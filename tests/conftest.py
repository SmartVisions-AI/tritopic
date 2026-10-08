"""Shared fixtures for TriTopic test suite."""

import numpy as np
import pytest

from tritopic import TriTopic


@pytest.fixture(scope="session")
def fake_documents():
    """30 short synthetic documents spanning 3 rough themes."""
    return [
        # Theme A: space / astronomy
        "NASA launched a new satellite into orbit around Mars",
        "The Hubble telescope captured images of a distant galaxy",
        "Astronomers discovered a new exoplanet in the habitable zone",
        "SpaceX successfully landed its booster after launch",
        "The International Space Station orbits Earth every 90 minutes",
        "Mars rover collected soil samples from the red planet",
        "A solar eclipse was visible across North America yesterday",
        "Scientists detected gravitational waves from merging black holes",
        "The Artemis mission aims to return humans to the Moon",
        "Jupiter has over 80 known moons orbiting the gas giant",
        # Theme B: sports / baseball
        "The pitcher threw a perfect game with 27 consecutive outs",
        "Home run records were broken during the baseball season",
        "The World Series attracted millions of viewers this year",
        "Spring training begins in February for all baseball teams",
        "The shortstop made an incredible diving catch in the ninth",
        "Baseball players negotiated a new collective bargaining agreement",
        "The minor league team won the championship last night",
        "Batting averages declined across the league this season",
        "The umpire called a controversial strike three to end the game",
        "Free agent signings dominated the baseball off-season news",
        # Theme C: medicine / health
        "A new vaccine was approved for the prevention of malaria",
        "Clinical trials showed promising results for the cancer drug",
        "Researchers published a study on the effects of sleep deprivation",
        "The hospital implemented a new protocol for emergency surgery",
        "Antibiotics resistance is a growing concern among health experts",
        "A breakthrough in gene therapy offers hope for rare diseases",
        "The flu season peaked earlier than expected this winter",
        "Doctors recommend regular exercise to prevent heart disease",
        "Mental health awareness campaigns are increasing worldwide",
        "A new diagnostic tool detects infections within minutes",
    ]


@pytest.fixture(scope="session")
def fake_embeddings(fake_documents):
    """Random embeddings matching fake_documents, deterministic seed."""
    rng = np.random.RandomState(42)
    n_docs = len(fake_documents)
    emb = rng.randn(n_docs, 64).astype(np.float32)
    # L2-normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / norms
    return emb


@pytest.fixture(scope="session")
def fitted_model(fake_documents, fake_embeddings):
    """A TriTopic model fitted on fake data (fast, no real embeddings)."""
    model = TriTopic(
        verbose=False,
        n_neighbors=5,
    )
    model.config.use_dim_reduction = False
    model.config.use_iterative_refinement = False
    model.config.n_consensus_runs = 3
    model.config.min_cluster_size = 2

    model.fit_transform(fake_documents, embeddings=fake_embeddings)
    return model
