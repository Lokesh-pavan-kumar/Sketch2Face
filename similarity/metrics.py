import torch


def top_1(embeddings1: torch.Tensor, embeddings2: torch.Tensor, normalize: bool = False):
    """
    :param embeddings1: list of image embeddings to be compared against
    :param embeddings2: list of image embeddings to be compared to
    :param normalize: boolean specifying whether or not to normalize the embeddings
    :return: for every tensor in embeddings2 returns the max_similar embedding from embeddings1, and also the accuracy
    """

    def norm(x: torch.Tensor):
        return x / torch.norm(x, dim=1, keepdim=True)

    n_samples = embeddings1.shape[0]
    if normalize:  # Normalize the tensors if needed
        embeddings1 = norm(embeddings1)
        embeddings2 = norm(embeddings2)
    count = 0
    indexes = []
    sim_values = []
    for idx, emb in enumerate(embeddings2):
        similarities = torch.mm(embeddings1, emb.T.unsqueeze(-1)).squeeze()  # Get the similarity values
        most_similar = torch.argmax(similarities).item()
        if most_similar == idx:
            count += 1
        indexes.append(most_similar)
        sim_values.append(similarities[most_similar])
    return count / n_samples, indexes, sim_values


def top_n(embeddings1: torch.Tensor, embeddings2: torch.Tensor, n: int = 3, normalize: bool = False):
    """
        :param embeddings1: list of image embeddings to be compared against
        :param embeddings2: list of image embeddings to be compared to
        :param n: the number of matches to search from
        :param normalize: boolean specifying whether or not to normalize the embeddings
        :return: for every tensor in embeddings2 returns the most similar 'n' embeddings from embeddings1,
        and also the accuracy
        """

    def norm(x: torch.Tensor):
        return x / torch.norm(x, dim=1, keepdim=True)

    n_samples = embeddings1.shape[0]
    if normalize:  # Normalize the tensors if needed
        embeddings1 = norm(embeddings1)
        embeddings2 = norm(embeddings2)
    count = 0
    indexes = []
    sim_values = []
    for idx, emb in enumerate(embeddings2):
        similarities = torch.mm(embeddings1, emb.T.unsqueeze(-1)).squeeze()  # Get the similarity values
        assert similarities.ndim == 1
        most_similar = torch.argsort(similarities, descending=True)[:n].tolist()
        if idx in most_similar:
            count += 1
        indexes.append(most_similar)
        sim_values.append(similarities[most_similar])
    return count / n_samples, indexes, sim_values
