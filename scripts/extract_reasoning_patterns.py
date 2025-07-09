import argparse
import json
import os
from pathlib import Path
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import re



def load_traces(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def extract_think_sentences(data):
    """
    Extracts the content between <think> and </think> tags from generated_text field.
    """
    reasoning_sentences = []
    for ex in data:
        text = ex["generated_text"]
        # Extract between <think> and </think>
        match = re.search(r'\bthink\b', text, re.IGNORECASE)
        if match:
            extracted = text[:match.start()].strip()
        else:
            # If no 'think' found, use whole text
            extracted = text.strip()
        # Split extracted text into sentences, keeping only sentences longer than 5 chars
        sentences = [s.strip() for s in re.split(r'[.?!]\s+', extracted) if len(s.strip()) > 5]
        reasoning_sentences.extend(sentences)
    return reasoning_sentences


def cluster_sentences(sentences, threshold=0.8, model_name="all-MiniLM-L6-v2"):
    """
    Groups similar sentences together using semantic similarity.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
    clusters = []
    visited = set()

    for i, emb in enumerate(embeddings):
        if i in visited:
            continue
        group = [sentences[i]]
        visited.add(i)
        sims = util.cos_sim(emb, embeddings)[0]
        for j in range(i + 1, len(sentences)):
            if j not in visited and sims[j] > threshold:
                group.append(sentences[j])
                visited.add(j)
        clusters.append(group)
    return clusters


def save_clusters(clusters, out_path="output/reasoning_clusters.json"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(clusters, f, indent=2)
    print(f"\n💾 Clustered reasoning sentences saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the reasoning trace JSON file")
    parser.add_argument("--output", type=str, default="output/reasoning_clusters.json", help="Path to save clustered output")
    parser.add_argument("--threshold", type=float, default=0.8, help="Cosine similarity threshold for clustering")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        return

    print(f"📥 Loading reasoning traces from {args.input}")
    data = load_traces(args.input)

    print("🔍 Extracting reasoning sentences...")
    all_sentences = extract_think_sentences(data)
    print(f"🧠 Collected {len(all_sentences)} reasoning sentences.")

    counter = Counter(all_sentences)

    print("🔗 Clustering similar sentences...")
    clustered = cluster_sentences(list(counter.keys()), threshold=args.threshold)

    cluster_dict = [{"count": len(c), "examples": c[:5]} for c in sorted(clustered, key=lambda x: -len(x))]

    print("\n🧩 Top 5 recurring reasoning sentence clusters:")
    for i, cluster in enumerate(cluster_dict[:5]):
        print(f"\nCluster #{i+1} ({cluster['count']} items):")
        for line in cluster["examples"]:
            print(f"  - {line}")
        if cluster['count'] > 5:
            print(f"  ... and {cluster['count'] - 5} more")

    save_clusters(cluster_dict, args.output)


if __name__ == "__main__":
    main()
