import os
import fasttext
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import seaborn as sns


def load_model(repo_id: str) -> fasttext.FastText._FastText:
    model_path = hf_hub_download(
        repo_id,
        filename="model.bin",
        cache_dir="fasttext/models",
    )

    return fasttext.load_model(model_path)


# repo_id="cis-lmu/glotlid"
repo_id = "facebook/fasttext-language-identification"
model = load_model(repo_id)


def identify_languages(title):
    predictions = model.predict(title, k=5)  # Get top 5 predictions
    labels = predictions[0]
    confidences = predictions[1]

    results = []
    for label, confidence in zip(labels, confidences):
        language = label.replace("__label__", "")
        results.append((language, confidence))

    return results


def plot_confidences(results):
    languages = [result[0] for result in results]
    confidences = [result[1] for result in results]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=confidences, y=languages)
    plt.xlabel("Confidence")
    plt.ylabel("Language")
    plt.title("Language Identification Confidence")

    return plt


def identify_and_plot(title):
    results = identify_languages(title)
    plot = plot_confidences(results)
    return results, plot


LID_EXAMPLES = [
    "ပိုၼ်းၵႅပ်ႈလိၵ်ႈလၢႆးတႆး ဢၼ်ၶူးပွင်သွၼ်လူင်လိၵ်ႈလၢႆးတႆး",
    "ယု၀တီဂျင်းဖောမယ်၊ ရှမ်းစာပေသမိုင်းနှင့်",
    "Hello World, မႂ်ႇသုင်ၶႃႈ",
]
