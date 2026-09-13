"""
Command-line trainer for the bilingual fake-news classifier.

Usage examples
--------------
# English (WELFake-style: columns "title", "text", "label")
python train.py --lang en --csv WELFake_Dataset.csv --text-col text \
    --title-col title --label-col label

# Arabic (columns "Article_content" or "title"+"text", "label")
python train.py --lang ar --csv arabic_fake_news_processed.csv \
    --text-col Article_content --label-col label

Trained artifacts are written to ./models/ and are picked up automatically
by app.py the next time it starts.
"""

import argparse
import sys

import pandas as pd

from src.model import train_and_save


def main():
    parser = argparse.ArgumentParser(description="Train the TF-IDF + Logistic Regression fake-news model.")
    parser.add_argument("--lang", choices=["en", "ar"], required=True, help="Which pipeline to train.")
    parser.add_argument("--csv", required=True, help="Path to the training CSV file.")
    parser.add_argument("--text-col", required=True, help="Column with the article body.")
    parser.add_argument("--title-col", default=None, help="Optional column with the article title (concatenated with --text-col).")
    parser.add_argument("--label-col", default="label", help="Column with the class label (default: 'label').")
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out test fraction (default: 0.2).")
    args = parser.parse_args()

    print(f"Loading {args.csv} ...")
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df):,} rows, columns: {list(df.columns)}")

    for col in [args.text_col, args.label_col] + ([args.title_col] if args.title_col else []):
        if col not in df.columns:
            print(f"ERROR: column '{col}' not found in {args.csv}", file=sys.stderr)
            sys.exit(1)

    df = df.dropna(subset=[args.text_col, args.label_col])
    df = df.drop_duplicates(subset=[args.text_col])

    def progress(msg, pct):
        print(f"[{pct*100:5.1f}%] {msg}")

    artifacts = train_and_save(
        df,
        lang=args.lang,
        label_col=args.label_col,
        text_col=args.text_col,
        title_col=args.title_col,
        test_size=args.test_size,
        progress_callback=progress,
    )

    print("\nTraining complete. Metrics on held-out test split:")
    for k, v in artifacts.metrics.items():
        print(f"  {k}: {v}")
    print(f"\nArtifacts saved under ./models/ for lang='{args.lang}'.")


if __name__ == "__main__":
    main()
