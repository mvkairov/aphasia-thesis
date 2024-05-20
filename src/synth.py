from common.utils import ObjectView, save_to_pkl
from train.synth import train_cvae, train_acgan
from common.tfutils import get_optimizer
from data.synth import get_dataset
from models.acgan import ACGAN
from models.cvae import CVAE

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import tensorflow as tf

from omegaconf import OmegaConf
import hydra

from dotenv import load_dotenv
import wandb
import os

val_models = {
    "mlp": MLPClassifier,
    "rf": RandomForestClassifier,
    # "gb": GradientBoostingClassifier
}


@hydra.main(version_base=None, config_path="../config", config_name="synth")
def my_app(cfg):
    cfg = ObjectView(OmegaConf.to_container(cfg))
    load_dotenv()
    tf.keras.utils.set_random_seed(int(os.environ.get("RANDOM_SEED")))

    train_ds, val_ds, input_dim, n_classes = get_dataset(**cfg.ds)
    cfg.model.input_dim = input_dim
    cfg.model.n_classes = n_classes
    model_id = cfg.model.id
    del cfg.model.id

    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(config=cfg, group=model_id, name=cfg.name, settings=wandb.Settings(start_method="fork"), mode="offline")
    
    cfg.optim.learning_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(cfg.optim.learning_rate, 1000)
    if model_id == "cvae":
        model = CVAE(**cfg.model)
        optim = get_optimizer(**cfg.optim)
        loss, score, pred, param = train_cvae(model, optim, train_ds, cfg.n_epochs, cfg.batch_size, val_models, val_ds, cfg.val_freq, cfg.name)
    elif model_id == "acgan":
        model = ACGAN(**cfg.model)
        gen_optim = get_optimizer(**cfg.optim)
        dis_optim = get_optimizer(**cfg.optim)
        loss, score, pred, param = train_acgan(model, gen_optim, dis_optim, train_ds, cfg.n_epochs, cfg.batch_size, val_models, val_ds, cfg.val_freq, cfg.name)
    wandb.finish()
    
    res = {
        "loss": loss,
        "score": score,
        "pred": pred,
        "param": param,
        "val_target": val_ds[1]
    }
    save_to_pkl(res, cfg.name, "results/synth-exp")

if __name__ == "__main__":
    my_app()
