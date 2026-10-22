from preprocessing import set_seed
def build_eff_model(seed=SEED):
    set_seed(seed)
    base = EfficientNetB0(weights="imagenet", include_top=False,
                          input_shape=(224, 224, 3))
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

model_eff = build_eff_model()
print("Total params:", model_eff.count_params())
print("Trainable params:",
      int(sum(np.prod(w.shape) for w in model_eff.trainable_weights)))

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=4, restore_best_weights=True
)

start_time = time.time()
history_eff = model_eff.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    callbacks=[early_stop]
)
training_time_eff = time.time() - start_time

h = history_eff.history
best_epoch = int(np.argmin(h["val_loss"]))   # epoch whose weights were restored

print(f"\nEpochs run: {len(h['loss'])}")
print(f"Training time: {training_time_eff / 60:.2f} minutes")
print(f"Restored (best val_loss) epoch: {best_epoch + 1}")
print(f"Val accuracy of restored model: {h['val_accuracy'][best_epoch] * 100:.2f}%")
print(f"(Highest val accuracy seen in any epoch: {max(h['val_accuracy']) * 100:.2f}%)")

validation_generator.reset()
val_probs = model_eff.predict(validation_generator, verbose=1)
val_pred  = val_probs.argmax(axis=1)
val_true  = validation_generator.classes

val_acc = accuracy_score(val_true, val_pred)
val_f1  = f1_score(val_true, val_pred, average="macro")

print(f"Validation Accuracy: {val_acc * 100:.2f}%")
print(f"Validation Macro-F1: {val_f1:.4f}\n")
print(classification_report(val_true, val_pred, target_names=class_names, digits=4))

model_eff.save_weights("mulberry_efficientnetb0_weights.h5")
print("Model weights saved (these are the restored best-epoch weights).")

EPOCHS = 30
BATCH = 32
SEEDS = [0, 1, 2]
PACE_START, PACE_RAMP = 0.30, 10     # start with 30% of data, reach 100% at epoch 11

def build_head(seed):
    """Same classifier head as the baseline: Dense(128) -> Dropout(0.3) -> Dense(10)."""
    set_seed(seed)
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(X_tr.shape[1],)),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m

def pacing(epoch):
    """Fraction of (easiest) training data used at a 0-indexed epoch."""
    return min(1.0, PACE_START + (1 - PACE_START) * epoch / PACE_RAMP)

print("Fraction of data used per epoch (curriculum):")
print([round(pacing(e), 2) for e in range(EPOCHS)])

q_va = pd.qcut(difficulty_va, 4, labels=False)   # 0 = easiest quartile, 3 = hardest (from Cell 20)
def run_experiment(mode, seed):
    model = build_head(seed)
    rng = np.random.default_rng(seed)
    rows, cum_time, samples_seen = [], 0.0, 0

    for ep in range(EPOCHS):
        if mode == "Random":
            idx = rng.permutation(len(y_tr))                        # all data, random order
        else:
            n = int(round(pacing(ep) * len(y_tr)))
            if mode == "Curriculum":
                idx = order_easy_bal[:n]                            # easiest n, strict easy -> hard
            else:  # "Curriculum-Shuffled"
                idx = rng.permutation(order_easy_bal[:n])           # easiest n, shuffled

        t0 = time.time()
        h = model.fit(X_tr[idx], y_tr[idx], batch_size=BATCH, epochs=1,
                      shuffle=False, verbose=0)
        cum_time += time.time() - t0                           # only training is timed
        samples_seen += len(idx)

        pred = model.predict(X_va, verbose=0).argmax(axis=1)
        rows.append(dict(method=mode, seed=seed, epoch=ep + 1,
                         n_used=len(idx), samples_seen=samples_seen,
                         train_loss=h.history["loss"][0],
                         train_acc=h.history["accuracy"][0],
                         val_acc=accuracy_score(y_va, pred),
                         val_f1=f1_score(y_va, pred, average="macro"),
                         cum_time=cum_time,
                         **{f"val_acc_Q{q+1}": (pred[q_va == q] == y_va[q_va == q]).mean()
                            for q in range(4)}))
    return model, pd.DataFrame(rows)

trained_models, histories = {}, []

for mode in ["Random", "Curriculum", "Curriculum-Shuffled"]:
    for seed in SEEDS:
        model, hist = run_experiment(mode, seed)
        trained_models[(mode, seed)] = model
        histories.append(hist)
        best = hist.loc[hist.val_acc.idxmax()]
        print(f"{mode:20s} seed={seed}  best val_acc={best.val_acc:.4f} (epoch {int(best.epoch)})  "
              f"final val_acc={hist.val_acc.iloc[-1]:.4f}  time={hist.cum_time.iloc[-1]:.1f}s")

all_hist = pd.concat(histories, ignore_index=True)
all_hist.to_csv("curriculum_vs_random_history.csv", index=False)


