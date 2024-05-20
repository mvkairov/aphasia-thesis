from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from tune.spaces import tab_spaces
from skopt import BayesSearchCV
import tensorflow as tf
import numpy as np
import wandb
import keras

from common.utils import sklearn_best_params, get_dict_of_lists, get_tqdm
from common.metrics import get_tune_metrics, get_val_metrics
from common.tfmetrics import reconstruction_loss

np.int = np.int32


def generate(generator, n_objects, class_id, classes=None):
    noise = np.random.randn(n_objects, generator.latent_dim)
    if classes is None:
        labels = np.tile(class_id, (n_objects, 1))
    else:
        labels = class_id * np.ones((n_objects, 1))
        labels = label_binarize(labels, classes=classes)
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)

    fake_data = generator.decode(noise, labels)
    return fake_data


def validate(generator, val_models, train_ds, val_ds, n_tune_iter=32, n_jobs=16, epoch=None):
    X_train, y_train = train_ds
    y_train = np.argmax(y_train, axis=1)
    y_unique, counts = np.unique(y_train, axis=0, return_counts=True)
    X_val, y_val = val_ds

    for i, c in zip(y_unique, counts):
        X_train = np.vstack((X_train, generate(generator, 2 * c, i, y_unique)))
        y_train = np.append(y_train, np.tile(i, (2 * c,)))

    tune_metrics = get_tune_metrics("clf", y_unique.shape[0])
    val_metrics = get_val_metrics("clf", False, "type", y_unique.shape[0])

    scores, predictions, params = {}, {}, {}
    for val_name, model in val_models.items():
        bayes_search = BayesSearchCV(
            model(), tab_spaces[val_name],
            scoring=tune_metrics, cv=StratifiedKFold(3),
            n_iter=n_tune_iter, n_jobs=n_jobs, n_points=n_jobs, 
        )
        bayes_search.fit(X_train, y_train)
        search_results = bayes_search.cv_results_
        params[val_name] = sklearn_best_params(search_results)
        y_pred = bayes_search.best_estimator_.predict(X_val)
        scores[val_name] = val_metrics(y_val, y_pred)
        predictions[val_name] = y_pred.tolist()

        for k, v in scores[val_name].items():
            print(f"{val_name}_{k} = {v:.5f}")
        print(flush=True)
        
        if wandb.run is not None:
            wandb.log({f"{val_name}_{k}": v for k, v in scores[val_name].items()}, step=epoch)
        
    return scores, predictions, params


def train_acgan(model, gen_opt, dis_opt, train_ds, n_epochs, batch_size, val_model=None, val_ds=None, val_freq=20, model_name="model"):
    dis_loss = gen_loss = 0
    dis_loss_fn = keras.losses.BinaryCrossentropy()
    aux_loss_fn = keras.losses.CategoricalCrossentropy()
    losses, scores, predictions, params = [], [], [], []
    best_loss = 1e6

    X_train, y_train = train_ds

    for epoch in range(1, n_epochs + 1):
        for i in get_tqdm()(range(X_train.shape[0] // batch_size)):
            X_batch = tf.convert_to_tensor(X_train[i * batch_size:(i + 1) * batch_size, :], dtype=tf.float32)
            y_batch = tf.convert_to_tensor(y_train[i * batch_size:(i + 1) * batch_size], dtype=tf.int32)
            labels_real = np.ones((batch_size, 1))
            labels_fake = np.zeros((batch_size, 1))
            
            noise = np.random.randn(batch_size, model.latent_dim)
            fake_rows = model.generate(noise, y_batch)
            # Update Discriminator with real and fake data: maximize log(D(x)) + log(1 - D(G(z)))
            with tf.GradientTape() as tape:
                # First, train on real images
                dis_logits_real, aux_logits_real = model.discriminate(X_batch)
                dis_loss_real = dis_loss_fn(labels_real, dis_logits_real)
                aux_loss_real = aux_loss_fn(y_batch, aux_logits_real)
                dis_loss_real = dis_loss_real + aux_loss_real
                # Then train on images generated from noise
                dis_logits_fake, aux_logits_fake = model.discriminate(fake_rows)
                dis_loss_fake = dis_loss_fn(labels_fake, dis_logits_fake)
                aux_loss_fake = aux_loss_fn(y_batch, aux_logits_fake)
                dis_loss_fake = dis_loss_fake + aux_loss_fake
                # Sum losses from real and fake images
                dis_loss = dis_loss_real + dis_loss_fake
            # Update discriminator weights
            dis_grads = tape.gradient(dis_loss, model.discriminator.trainable_weights)
            dis_opt.apply_gradients(zip(dis_grads, model.discriminator.trainable_weights))

            # Update Generator: maximize log(D(G(z)))
            with tf.GradientTape() as tape:
                fake_rows = model.generate(noise, y_batch)
                dis_logits_gen, aux_logits_gen = model.discriminate(fake_rows)
                fake_loss = dis_loss_fn(labels_real, dis_logits_gen)
                aux_loss = aux_loss_fn(y_batch, aux_logits_gen)
                gen_loss = fake_loss + aux_loss
            # Update generator weights
            gen_grads = tape.gradient(gen_loss, model.generator.trainable_weights)
            gen_opt.apply_gradients(zip(gen_grads, model.generator.trainable_weights))
        
        print(f"EPOCH {epoch}. Loss: D={dis_loss:.5f}, G={gen_loss:.5f}\n")
        losses.append((dis_loss.numpy().tolist(), gen_loss.numpy().tolist()))
        
        if best_loss > tf.math.reduce_sum(dis_loss + gen_loss):
            best_loss = tf.math.reduce_sum(dis_loss + gen_loss)
            model.save(f"results/checkpoints/{model_name}.keras")
        
        if wandb.run is not None:
            wandb.log({
                "dis_loss": dis_loss,
                "gen_loss": gen_loss
            }, step=epoch)
            
        if val_model is not None and (epoch % val_freq == 0 or epoch == n_epochs):
            score, pred, param = validate(model, val_model, train_ds, val_ds, epoch=epoch)
            scores.append(score)
            predictions.append(pred)
            params.append(param)

    scores = get_dict_of_lists(scores)
    predictions = get_dict_of_lists(predictions)
    params = get_dict_of_lists(params)
    return losses, scores, predictions, params


def train_cvae(model, optim, train_ds, n_epochs, batch_size, val_models=None, val_ds=None, val_freq=20, model_name="model"):
    loss = 0
    losses, scores, predictions, params = [], [], [], []
    X_train, y_train = train_ds
    best_loss = 1e6

    for epoch in range(1, n_epochs + 1):
        for i in get_tqdm()(range(X_train.shape[0] // batch_size)):
            X_batch = tf.convert_to_tensor(X_train[i * batch_size:(i + 1) * batch_size, :], dtype=tf.float32)
            y_batch = tf.convert_to_tensor(y_train[i * batch_size:(i + 1) * batch_size], dtype=tf.int32)

            with tf.GradientTape() as tape:
                recon_x, mean, log_var = model(X_batch, y_batch)
                loss = reconstruction_loss(X_batch, recon_x, mean, log_var)
            grads = tape.gradient(loss, model.trainable_weights)
            optim.apply_gradients(zip(grads, model.trainable_weights))
        
        print(f"EPOCH {epoch}. Loss = {loss:.5f}\n")
        losses.append(loss.numpy().tolist())
        if best_loss > tf.math.reduce_sum(loss):
            best_loss = tf.math.reduce_sum(loss)
            model.save(f"results/checkpoints/{model_name}.keras")
        
        if wandb.run is not None:
            wandb.log({
                "loss": loss,
            }, step=epoch)
        
        if val_models is not None and (epoch % val_freq == 0 or epoch == n_epochs):
            score, pred, param = validate(model, val_models, train_ds, val_ds, epoch=epoch)
            scores.append(score)
            predictions.append(pred)
            params.append(param)

    scores = get_dict_of_lists(scores)
    predictions = get_dict_of_lists(predictions)
    params = get_dict_of_lists(params)
    return losses, scores, predictions, params
