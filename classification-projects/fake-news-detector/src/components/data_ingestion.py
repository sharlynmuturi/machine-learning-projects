import pandas as pd
import json
from pathlib import Path
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)


class DataIngestion:
    def __init__(
        self,
        raw_data_dir="data/raw",
        processed_data_dir="data/processed",
        output_file="fake_news_full.csv"
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.output_path = self.processed_data_dir / output_file


    # LIAR DATASET
    def _load_liar(self) -> pd.DataFrame:
        try:
            logger.info("Loading LIAR dataset")

            cols = [
                "id", "label", "statement", "subject", "speaker",
                "job", "state", "party", "barely_true",
                "false", "half_true", "mostly_true",
                "pants_fire", "context"
            ]

            dfs = []
            liar_path = self.raw_data_dir / "liar"

            for split in ["train.tsv", "valid.tsv", "test.tsv"]:
                df = pd.read_csv(liar_path / split, sep="\t", header=None, names=cols)

                df["label"] = df["label"].apply(
                    lambda x: 1 if x in ["true", "mostly-true"] else 0
                )

                dfs.append(df)

            liar = pd.concat(dfs, ignore_index=True)

            return pd.DataFrame({
                "title": None,
                "text": liar["statement"],
                "source": liar["speaker"],
                "date": None,
                "dataset": "LIAR",
                "label": liar["label"]
            })

        except Exception as e:
            raise CustomException(e)

    # ISOT DATASET
    def _load_isot(self) -> pd.DataFrame:
        try:
            logger.info("Loading ISOT dataset")

            isot_path = self.raw_data_dir / "isot"
            fake = pd.read_csv(isot_path / "Fake.csv")
            true = pd.read_csv(isot_path / "True.csv")

            fake["label"] = 0
            true["label"] = 1

            isot = pd.concat([fake, true], ignore_index=True)

            return pd.DataFrame({
                "title": isot["title"],
                "text": isot["text"],
                "source": None,
                "date": isot.get("date"),
                "dataset": "ISOT",
                "label": isot["label"]
            })

        except Exception as e:
            raise CustomException(e)

    # FakeNewsNet DATASET
    def _load_fakenewsnet(self) -> pd.DataFrame:
        import pandas as pd
        import os
    
        records = []
        fakenewsnet_path = self.raw_data_dir / "fakenewsnet"
    
        datasets = [
            ("politifact_fake.csv", 0, "FakeNewsNet-Politifact"),
            ("politifact_real.csv", 1, "FakeNewsNet-Politifact"),
            ("gossipcop_fake.csv", 0, "FakeNewsNet-GossipCop"),
            ("gossipcop_real.csv", 1, "FakeNewsNet-GossipCop"),
        ]
    
        for filename, label, dataset_name in datasets:
            file_path = fakenewsnet_path / filename
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing file: {file_path}")
    
            df = pd.read_csv(file_path)
    
            # Directly use 'title' as text
            for _, row in df.iterrows():
                records.append({
                    "title": row["title"],
                    "text": row["title"], 
                    "source": row.get("news_url", None),
                    "date": None, 
                    "dataset": dataset_name,
                    "label": label
                })
    
        return pd.DataFrame(records)

    # MASTER INGESTION
    def initiate_data_ingestion(self) -> str:
        try:
            logger.info("Starting full data ingestion pipeline")

            liar_df = self._load_liar()
            isot_df = self._load_isot()
            fakenewsnet_df = self._load_fakenewsnet()

            df = pd.concat(
                [liar_df, isot_df, fakenewsnet_df],
                ignore_index=True
            )

            df = df.dropna(subset=["text"])
            df["text"] = df["text"].astype(str)

            self.processed_data_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.output_path, index=False)

            logger.info(
                f"Data ingestion completed successfully. "
                f"Total samples: {len(df)}"
            )

            return str(self.output_path)

        except Exception as e:
            logger.error("Data ingestion failed")
            raise CustomException(e)