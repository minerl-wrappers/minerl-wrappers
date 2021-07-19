import minerl
import numpy as np
import tqdm
from sklearn.cluster import KMeans


def generate_kmeans(env_id, n_clusters, random_state):
    print(f"loading data...")
    dat = minerl.data.make(env_id)
    act_vectors = []
    for _, act, _, _, _ in tqdm.tqdm(
        dat.batch_iter(
            batch_size=16,
            seq_len=32,
            num_epochs=1,
            preload_buffer_size=32,
            seed=random_state,
        )
    ):
        act_vectors.append(act["vector"])
    acts = np.concatenate(act_vectors).reshape(-1, 64)
    print(f"loading data... done.")
    print(f"executing keamns...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(acts)
    print(f"executing keamns... done.")
    return kmeans


if __name__ == "__main__":
    means = generate_kmeans(
        env_id="MineRLObtainDiamondDenseVectorObf-v0", n_clusters=30, random_state=1337
    )
    np.save("./data/means.npy", means.cluster_centers_)