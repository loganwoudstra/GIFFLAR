from glycowork.ml.model_training import training_setup, train_model
from glycowork.ml.models import prep_model


def train(hidden_dim: int = 128):
    model = prep_model("SweetNet", None, hidden_dim=hidden_dim)
    optimizer, lr, criterion = training_setup(model, 0.0001)
    m = train_model(model, None, criterion, optimizer, lr, 100, 20, mode="classification", mode2="multi")
