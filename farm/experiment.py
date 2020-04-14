import logging
from pathlib import Path

from farm.data_handler.data_silo import DataSilo
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.prediction_head import PredictionHead
from farm.modeling.tokenization import Tokenizer
from farm.data_handler.processor import Processor
from farm.train import Trainer, EarlyStopping
from farm.utils import set_all_seeds, initialize_device_settings
from farm.utils import MLFlowLogger as MlLogger
from farm.file_utils import read_config, unnestConfig
from farm.infer import Inferencer
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.WARNING#ERROR #CRITICAL
    #level=logging.INFO,
)


def load_experiments(file):
    args = read_config(file, flattend=False)
    experiments = unnestConfig(args, flattened=False)
    return experiments


def run_experiment(args):
    logger.info(
        "\n***********************************************"
        f"\n************* Experiment: {args.task.name} ************"
        "\n************************************************"
    )
    ml_logger = MlLogger(tracking_uri=args.logging.mlflow_url)
    ml_logger.init_experiment(
        experiment_name=args.logging.mlflow_experiment,
        run_name=args.logging.mlflow_run_name,
        nested=args.logging.mlflow_nested,
    )

    validate_args(args)
    distributed = bool(args.general.local_rank != -1)

    # Init device and distributed settings
    device, n_gpu = initialize_device_settings(
        use_cuda=args.general.cuda,
        local_rank=args.general.local_rank,
        use_amp=args.general.use_amp,
    )

    args.parameter.batch_size = int(
        args.parameter.batch_size // args.parameter.gradient_accumulation_steps
    )
    # if n_gpu > 1:
    #     args.parameter.batch_size = args.parameter.batch_size * n_gpu
    set_all_seeds(args.general.seed)

    # Prepare Data
    tokenizer = Tokenizer.load(
        args.parameter.model, do_lower_case=args.parameter.lower_case
    )
    processor = Processor.load(
        tokenizer=tokenizer,
        max_seq_len=args.parameter.max_seq_len,
        data_dir=Path(args.general.data_dir),
        **args.task.toDict(),  # args is of type DotMap and needs conversion to std python dicts
    )

    data_silo = DataSilo(
        processor=processor,
        batch_size=args.parameter.batch_size,
        distributed=distributed,
    )

    class_weights = None
    if args.parameter.balance_classes:
        task_names = list(processor.tasks.keys())
        if len(task_names) > 1:
            raise NotImplementedError(f"Balancing classes is currently not supported for multitask experiments. Got tasks:  {task_names} ")
        class_weights = data_silo.calculate_class_weights(task_name=task_names[0])

    model = get_adaptive_model(
        lm_output_type=args.parameter.lm_output_type,
        prediction_heads=args.parameter.prediction_head,
        layer_dims=args.parameter.layer_dims,
        model=args.parameter.model,
        device=device,
        class_weights=class_weights,
        embeds_dropout_prob=args.parameter.embeds_dropout_prob,
    )

    # Init optimizer
    optimizer_opts = dict(args.optimizer.optimizer_opts) if args.optimizer.optimizer_opts else None
    schedule_opts = dict(args.optimizer.schedule_opts) if args.optimizer.schedule_opts else None
    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=args.optimizer.learning_rate,
        schedule_opts=schedule_opts,
        optimizer_opts=optimizer_opts,
        use_amp=args.general.use_amp,
        n_batches=len(data_silo.loaders["train"]),
        grad_acc_steps=args.parameter.gradient_accumulation_steps,
        n_epochs=args.parameter.epochs,
        device=device
    )
    # An early stopping instance can be used to save the model that performs best on the dev set
    # according to some metric and stop training when no improvement is happening for some iterations.
    earlystopping = EarlyStopping(
        metric="f1_weighted", mode="max",  # use f1_macro from the dev evaluator of the trainer
        #metric="loss", mode="min",   # use loss from the dev evaluator of the trainer
        save_dir=Path("saved_models/bert-trac2020/"),  # where to save the best model
        patience=2    # number of evaluations to wait for improvement before terminating the training
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=args.parameter.epochs,
        n_gpu=n_gpu,
        grad_acc_steps=args.parameter.gradient_accumulation_steps,
        use_amp=args.general.use_amp,
        local_rank=args.general.local_rank,
        lr_schedule=lr_schedule,
        evaluate_every=args.logging.eval_every,
        device=device,
	early_stopping=earlystopping
    )

    model = trainer.train()

    model_name = (
        f"{model.language_model.name}-{model.language_model.language}-{args.task.name}"
    )
    processor.save(Path(f"{args.general.output_dir}/{model_name}"))
    model.save(Path(f"{args.general.output_dir}/{model_name}"))
    
    # inference for TRAc2020
    language = "eng" # eng iben hin
    option = "" # "_sep_emoji" or ""
    submission = "_dev" # "_dev" or "_test"
    
    df = pd.read_csv("data/trac2020/trac2_"+language+submission+option+".csv")
    df = df.rename({'Text': 'text'}, axis='columns')
    df = df.filter(['text'])
    basic_texts = df.to_dict('records')
    print("LOADING INFERENCER FROM BEST MODEL DURING TRAINING")
    model = Inferencer.load(model_name_or_path=earlystopping.save_dir, return_class_probs=True, batch_size=48, gpu=True)
    result = model.inference_from_dicts(dicts=basic_texts, max_processes=40)
    print("APPLICATION ON BEST MODEL")
    result = [x['predictions'] for x in result]
    probabilities = [x['probability'] for x in [item for sublist in result for item in sublist]]
    df_probabilities = pd.DataFrame.from_records(probabilities)
    df_probabilities.to_csv("data/trac2020/trac2_"+language+submission+"_prediction-"+str(model_name)+"-"+str(args.general.seed)+".csv",index=False)
    
    # if we make predictions for the development/validation set, we can evaluate the predictions.
    if submission == "_dev":
        df_dev = pd.read_csv("data/trac2020/trac2_"+language+"_dev"+option+".csv")
        df_dev = df_dev.filter(['Sub-task A'])
        label_to_index = {'NAG':0, 'CAG':1, 'OAG':2}
        df_dev = df_dev.applymap(lambda x: label_to_index[x])
        y_true = df_dev.values
        y_pred = df_probabilities.idxmax(axis = 1, skipna = True)
        target_names = ['NAG', 'CAG', 'OAG']
        print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
    
        #df_probabilities['group'] = np.arange(len(df_probabilities))
        #dfs = [df_probabilities, df_probabilities]
        #final = pd.concat(dfs, ignore_index=True)
        #print(final.groupby('group').mean())
    ml_logger.end_run()

def get_adaptive_model(
    lm_output_type,
    prediction_heads,
    layer_dims,
    model,
    device,
    embeds_dropout_prob,
    class_weights=None,
):
    parsed_lm_output_types = lm_output_type.split(",")
    language_model = LanguageModel.load(model)
    
    initialized_heads = []
    for head_name in prediction_heads.split(","):
        initialized_heads.append(
            PredictionHead.create(
                prediction_head_name=head_name,
                layer_dims=layer_dims,
                class_weights=class_weights,
            )
        )

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=initialized_heads,
        embeds_dropout_prob=embeds_dropout_prob,
        lm_output_types=parsed_lm_output_types,
        device=device,
    )
    return model


def validate_args(args):
    if not args.task.do_train and not args.task.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.parameter.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.parameter.gradient_accumulation_steps
            )
        )


def save_model():
    raise NotImplementedError


def load_model():
    raise NotImplementedError
