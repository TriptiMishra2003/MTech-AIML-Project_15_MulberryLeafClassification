plt.figure(figsize=(12, 6))
x = np.arange(len(df))

plt.bar(x, df["Training Images"], width=0.4, label="Training")
plt.bar(x + 0.4, df["Testing Images"], width=0.4, label="Testing")

plt.xlabel("Classes")
plt.ylabel("Number of Images")
plt.title("Training and Testing Images per Class")
plt.xticks(x + 0.2, df["Class"], rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
x = np.arange(len(df_split))
width = 0.25

plt.bar(x - width, df_split["Training"], width, label="Training")
plt.bar(x, df_split["Validation"], width, label="Validation")
plt.bar(x + width, df_split["Testing"], width, label="Testing")

plt.xlabel("Classes")
plt.ylabel("Number of Images")
plt.title("Class-wise Distribution of Training, Validation and Testing Images")
plt.xticks(x, df_split["Class"], rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.show()

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

epochs_range = range(1, len(h["loss"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs_range, h["accuracy"], label="Training Accuracy")
axes[0].plot(epochs_range, h["val_accuracy"], label="Validation Accuracy")
axes[0].set_title("EfficientNetB0: Training and Validation Accuracy")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")

axes[1].plot(epochs_range, h["loss"], label="Training Loss")
axes[1].plot(epochs_range, h["val_loss"], label="Validation Loss")
axes[1].set_title("EfficientNetB0: Training and Validation Loss")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")

for ax in axes:
    ax.axvline(best_epoch + 1, color="gray", linestyle="--", label="Restored epoch")
    ax.legend()

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].hist(difficulty, bins=50, color="steelblue", edgecolor="white")
axes[0].axvline(0, color="red", linestyle="--", label="margin = 0")
axes[0].set_title("Distribution of Sample Difficulty (−margin)")
axes[0].set_xlabel("Difficulty"); axes[0].set_ylabel("Number of images")
axes[0].legend()

axes[1].boxplot([difficulty[y_tr == c] for c in range(num_classes)],
                labels=[n.split(" ", 1)[1] for n in class_names])
axes[1].set_title("Difficulty per Class")
axes[1].set_ylabel("Difficulty")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

diff_table = pd.DataFrame({
    "Class": class_names,
    "Mean difficulty": [difficulty[y_tr == c].mean() for c in range(num_classes)],
    "% margin < 0": [(margin[y_tr == c] < 0).mean() * 100 for c in range(num_classes)],
}).sort_values("Mean difficulty", ascending=False)
print(diff_table.round(4).to_string(index=False))

short = [n.split(" ", 1)[1] for n in class_names]

def show_grid(indices, title):
    fig, axes = plt.subplots(1, 8, figsize=(20, 3.5))
    for ax, i in zip(axes, indices):
        ax.imshow(Image.open(tr_files[i]))
        ax.set_title(f"{short[y_tr[i]]}\nnear: {short[closest_other_cls[i]]}\n"
                     f"margin={margin[i]:.3f}", fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()

show_grid(order_easy[:8], "8 Easiest Training Images")
show_grid(order_easy[-8:][::-1], "8 Hardest Training Images")

Z_va = l2_normalize(X_va)
sims_va = Z_va @ centroids.T
iv = np.arange(len(y_va))
own_va = sims_va[iv, y_va]
oth_va = sims_va.copy(); oth_va[iv, y_va] = -np.inf
difficulty_va = -(own_va - oth_va.max(axis=1))

quartile = pd.qcut(difficulty_va, 4, labels=["Q1 (easiest)", "Q2", "Q3", "Q4 (hardest)"])
check = pd.DataFrame({"quartile": quartile, "correct": (val_pred == y_va)})
print("Baseline validation accuracy by difficulty quartile:")
print((check.groupby("quartile")["correct"].mean() * 100).round(2))

n30 = int(0.30 * len(y_tr))
comp = pd.DataFrame({
    "Class": short,
    "Global sort": np.bincount(y_tr[order_easy[:n30]], minlength=num_classes),
    "Class-balanced": np.bincount(y_tr[order_easy_bal[:n30]], minlength=num_classes),
    "Full train set": np.bincount(y_tr, minlength=num_classes),
})
print(f"Class composition of the easiest 30% ({n30} images):")
print(comp.to_string(index=False))

x = np.arange(num_classes)
plt.figure(figsize=(12, 5))
plt.bar(x - 0.2, comp["Global sort"], 0.4, label="Global sort")
plt.bar(x + 0.2, comp["Class-balanced"], 0.4, label="Class-balanced sort")
plt.xticks(x, short, rotation=45, ha="right")
plt.ylabel("Images in easiest 30%")
plt.title("Class Composition of the Easiest 30% of Training Data")
plt.legend()
plt.tight_layout()
plt.show()

random_best_mean = all_hist[all_hist.method == "Random"].groupby("seed").val_acc.max().mean()
TARGET = round(0.95 * random_best_mean, 4)
print(f"Target validation accuracy = 95% of Random's mean best = {TARGET:.4f}\n")

def first_hit(df, target):
    hit = df[df.val_acc >= target]
    return (int(hit.epoch.iloc[0]), hit.cum_time.iloc[0]) if len(hit) else (np.nan, np.nan)

rows = []
for (mode, seed), df in all_hist.groupby(["method", "seed"]):
    ep_t, time_t = first_hit(df, TARGET)
    rows.append(dict(Method=mode, Seed=seed,
                     Best_Val_Acc=df.val_acc.max(),
                     Final_Val_Acc=df.val_acc.iloc[-1],
                     Best_Macro_F1=df.val_f1.max(),
                     Final_Macro_F1=df.val_f1.iloc[-1],
                     Epochs_to_Target=ep_t,
                     Time_to_Target_s=time_t,
                     Total_Train_Time_s=df.cum_time.iloc[-1]))

per_seed = pd.DataFrame(rows)
print("Per-seed results:")
print(per_seed.round(4).to_string(index=False))

summary = per_seed.drop(columns="Seed").groupby("Method").agg(["mean", "std"]).round(4)
print("\nMean ± std over 3 seeds:")
print(summary.T.to_string())
methods = ["Random", "Curriculum", "Curriculum-Shuffled"]
colors = {"Random": "tab:blue", "Curriculum": "tab:red", "Curriculum-Shuffled": "tab:green"}

num_cols = all_hist.select_dtypes(include="number").columns.drop("epoch")
g = all_hist.groupby(["method", "epoch"])[num_cols]
mean_h = g.mean().reset_index()
std_h = g.std().reset_index()

def plot_metric(ax, col, title, ylabel):
    for m in methods:
        mu, sd = mean_h[mean_h.method == m], std_h[std_h.method == m]
        ax.plot(mu.epoch, mu[col], label=m, color=colors[m])
        ax.fill_between(mu.epoch, mu[col] - sd[col], mu[col] + sd[col],
                        alpha=0.2, color=colors[m])
    ax.axvline(PACE_RAMP + 1, ls=":", color="gray", label="Curriculum reaches 100% data")
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)

fig, axes = plt.subplots(1, 3, figsize=(19, 5))
plot_metric(axes[0], "val_acc", "Validation Accuracy", "Accuracy")
axes[0].axhline(TARGET, ls="--", color="black", label=f"Target ({TARGET:.3f})")
plot_metric(axes[1], "val_f1", "Validation Macro-F1", "Macro-F1")
plot_metric(axes[2], "train_loss", "Training Loss (on the samples used)", "Loss")
for ax in axes: ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
for m in methods:
    mu = mean_h[mean_h.method == m]
    ax.plot(mu.samples_seen, mu.val_acc, label=m, color=colors[m])
ax.axhline(TARGET, ls="--", color="black", label="Target")
ax.set_title("Validation Accuracy vs Number of Training Samples Seen")
ax.set_xlabel("Cumulative training samples seen"); ax.set_ylabel("Validation accuracy")
ax.legend(); plt.tight_layout(); plt.show()

fig, axes = plt.subplots(1, 4, figsize=(22, 4.5), sharey=True)
qnames = ["Q1 (easiest)", "Q2", "Q3", "Q4 (hardest)"]
for q, ax in enumerate(axes):
    col = f"val_acc_Q{q+1}"
    for m in methods:
        mu = mean_h[mean_h.method == m]
        ax.plot(mu.epoch, mu[col], label=m, color=colors[m])
    ax.axvline(PACE_RAMP + 1, ls=":", color="gray")
    ax.set_title(f"Validation accuracy — {qnames[q]}"); ax.set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy"); axes[0].legend(fontsize=8)
plt.tight_layout(); plt.show()

print("Mean per-quartile validation accuracy at selected epochs:")
cols = [f"val_acc_Q{q+1}" for q in range(4)]
print(mean_h[mean_h.epoch.isin([1, 3, 6, 11, 30])][["method", "epoch"] + cols]
      .round(3).to_string(index=False))

test_rows = []
for (mode, seed), model in trained_models.items():
    pred = model.predict(X_te, verbose=0).argmax(axis=1)
    test_rows.append(dict(Method=mode, Seed=seed,
                          Test_Acc=accuracy_score(y_te, pred),
                          Test_Macro_F1=f1_score(y_te, pred, average="macro")))
test_df = pd.DataFrame(test_rows)

print("Test results per seed:")
print(test_df.round(4).to_string(index=False))

test_summary = test_df.drop(columns="Seed").groupby("Method").agg(["mean", "std"]).round(4)
print("\nTest results, mean ± std over 3 seeds:")
print(test_summary.to_string())

# Reference: the image-level EfficientNetB0 baseline (with augmentation) from Cell 11
te_plain.reset()
base_pred = model_eff.predict(te_plain, verbose=0).argmax(axis=1)
print(f"\nReference (image-level baseline, Cell 11): "
      f"Test Acc = {accuracy_score(y_te, base_pred):.4f}, "
      f"Test Macro-F1 = {f1_score(y_te, base_pred, average='macro'):.4f}")

from sklearn.metrics import ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 3, figsize=(24, 7))
for ax, m in zip(axes, methods):
    pred = trained_models[(m, 0)].predict(X_te, verbose=0).argmax(axis=1)
    ConfusionMatrixDisplay.from_predictions(
        y_te, pred, display_labels=short, normalize="true",
        values_format=".2f", cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"{m} (seed 0) — Test, row-normalised")
    ax.tick_params(axis="x", rotation=60)
plt.tight_layout(); plt.show()

per_class = {}
for m in methods:
    f1s = [f1_score(y_te, trained_models[(m, s)].predict(X_te, verbose=0).argmax(axis=1),
                    average=None) for s in SEEDS]
    per_class[m] = np.mean(f1s, axis=0)

per_class_df = pd.DataFrame(per_class, index=short).round(4)
per_class_df["Test support"] = np.bincount(y_te, minlength=num_classes)
print("Per-class test F1 (mean over 3 seeds):")
print(per_class_df.to_string())

