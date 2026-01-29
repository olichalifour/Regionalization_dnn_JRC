import os
import pickle
import time
from data.dataloader import get_dataloaders

def load_or_prepare_data(config, plot_dir):
    """
    Load full preprocessed data (including DataLoaders) from pickle if available.
    If not, run get_dataloaders() and pickle the full dictionary for next time.
    """
    save_path = os.path.join(plot_dir, "preprocessed_data.pkl")

    if os.path.exists(save_path):
        print(f"�� Found preprocessed dataset at {save_path}. Loading pickle...")
        t0 = time.perf_counter()
        with open(save_path, "rb") as f:
            data = pickle.load(f)
        print(f"✅ Loaded in {time.perf_counter() - t0:.2f} seconds.")
        return data

    else:
        print("�� No preprocessed data found. Building dataset with dataloader...")
        t0 = time.perf_counter()
        data = get_dataloaders(config["data"], config["training"], plot_dir)
        print(f"✅ Data prepared in {time.perf_counter() - t0:.2f} seconds.")

        print(f"�� Saving full data dictionary to {save_path}...")
        with open(save_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("✅ Pickle saved successfully.")

        return data