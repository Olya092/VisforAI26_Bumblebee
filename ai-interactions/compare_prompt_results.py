import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULT_FILES = {
    "Baseline": ROOT / "baseline_results.csv",
    "Wording change": ROOT / "wording_changes_results.csv",
    "Prompt ensemble": ROOT / "prompt_ensemble_results.csv",
}


def load_results(path):
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


results = {
    name: load_results(path)
    for name, path in RESULT_FILES.items()
}

baseline = results["Baseline"]
for name, rows in results.items():
    if len(rows) != len(baseline):
        raise ValueError(f"{name} has a different number of test rows.")
    for baseline_row, row in zip(baseline, rows):
        if (
            baseline_row["image_id"] != row["image_id"]
            or baseline_row["actual_class"] != row["actual_class"]
        ):
            raise ValueError(f"{name} does not match the baseline test rows.")


def overall_accuracy(rows):
    correct = sum(row["actual_class"] == row["predicted_class"] for row in rows)
    return correct / len(rows) * 100


accuracies = {
    name: overall_accuracy(rows)
    for name, rows in results.items()
}

classes = sorted({row["actual_class"] for row in baseline})
class_accuracies = {}
for name, rows in results.items():
    class_accuracies[name] = {}
    for class_name in classes:
        class_rows = [row for row in rows if row["actual_class"] == class_name]
        class_accuracies[name][class_name] = overall_accuracy(class_rows)

baseline_name = "Baseline"
improved_name = "Prompt ensemble"
class_deltas = {
    class_name: (
        class_accuracies[improved_name][class_name]
        - class_accuracies[baseline_name][class_name]
    )
    for class_name in classes
}
sorted_classes = sorted(classes, key=lambda class_name: class_deltas[class_name])
delta_values = [class_deltas[class_name] for class_name in sorted_classes]
colors = ["#d95f5f" if delta < 0 else "#3b9b70" for delta in delta_values]

fig, (accuracy_ax, classes_ax) = plt.subplots(
    1,
    2,
    figsize=(16, max(10, len(classes) * 0.19)),
    gridspec_kw={"width_ratios": [1, 2.4]},
)

names = list(accuracies)
values = [accuracies[name] for name in names]
bars = accuracy_ax.bar(
    names,
    values,
    color=["#617a9f", "#e7a34b", "#3b9b70"],
)
accuracy_ax.set_ylim(0, 100)
accuracy_ax.set_ylabel("Accuracy (%)")
accuracy_ax.set_title("Overall accuracy on the same 10,000 test images")
accuracy_ax.tick_params(axis="x", rotation=20)
accuracy_ax.grid(axis="y", linestyle="--", alpha=0.35)
for bar, value in zip(bars, values):
    accuracy_ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
    )

classes_ax.barh(sorted_classes, delta_values, color=colors)
classes_ax.axvline(0, color="black", linewidth=0.8)
classes_ax.set_xlabel("Change in class accuracy (percentage points)")
classes_ax.set_title("Per-class change: prompt ensemble minus baseline")
classes_ax.grid(axis="x", linestyle="--", alpha=0.35)
classes_ax.tick_params(axis="y", labelsize=7)

overall_change = accuracies[improved_name] - accuracies[baseline_name]
fig.suptitle(
    f"Prompt improvement comparison — overall change: {overall_change:+.2f} percentage points",
    fontsize=15,
)
fig.tight_layout(rect=(0, 0, 1, 0.98))

output_path = ROOT / "prompt_results_comparison.png"
fig.savefig(output_path, dpi=180, bbox_inches="tight")
print(f"Saved chart: {output_path}")
print("\nOverall accuracy:")
for name, accuracy in accuracies.items():
    print(f"{name}: {accuracy:.2f}%")
print(f"\nPrompt ensemble vs baseline: {overall_change:+.2f} percentage points")
print(
    "Classes improved:",
    sum(delta > 0 for delta in class_deltas.values()),
    "| unchanged:",
    sum(delta == 0 for delta in class_deltas.values()),
    "| declined:",
    sum(delta < 0 for delta in class_deltas.values()),
)
print("\nLargest per-class gains:")
for class_name in sorted(classes, key=lambda item: class_deltas[item], reverse=True)[:5]:
    print(f"{class_name}: {class_deltas[class_name]:+.2f} points")
print("\nLargest per-class declines:")
for class_name in sorted(classes, key=lambda item: class_deltas[item])[:5]:
    print(f"{class_name}: {class_deltas[class_name]:+.2f} points")
