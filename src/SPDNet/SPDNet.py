from spd_learn.modules import BiMap, CovLayer, LogEig, ReEig, SPDBatchNormMeanVar
import torch
import torch.nn as nn


class SPDNetBatchNorm(nn.Module):

    def __init__(
        self,
        input_type="raw",
        cov_method=covariance,
        subspacedim=None,
        threshold=1e-4,
        upper=True,
        n_chans=None,
        n_outputs=None,
        bn=None,
    ):
        super().__init__()

        if subspacedim is None:
            warn(
                "subspacedim is None, using the default value of "
                "the number of channels",
                UserWarning,
            )
            subspacedim = n_chans

        if input_type == "raw":
            self.cov = CovLayer(method=cov_method)
        elif input_type == "cov":
            self.cov = nn.Identity()

        self.bimap = BiMap(n_chans, subspacedim)
        self.bn = bn if bn is not None else nn.Identity()
        self.reeig = ReEig(threshold)
        self.logeig = LogEig(upper=upper)
        self.len_last_layer = (
            subspacedim * (subspacedim + 1) // 2 if upper else subspacedim ** 2
        )
        self.classifier = nn.Linear(self.len_last_layer, n_outputs)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.cov(X)
        X = self.bimap(X)
        X = self.bn(X)   # FIX: no longer needs a None-check
        X = self.reeig(X)
        X = self.logeig(X)
        X = self.classifier(X)
        return X



def train_model(model, X_train, y_train, X_test, y_test, epochs=200, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(X_test).argmax(1) == y_test).float().mean().item()
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Test Acc: {acc:.4f}")
