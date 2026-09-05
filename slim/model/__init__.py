__all__ = ["SLIMModel", "SLIMTransformer"]


def __getattr__(name):
    if name == "SLIMModel":
        from .slim_model import SLIMModel

        return SLIMModel
    if name == "SLIMTransformer":
        from .slim_transformer import SLIMTransformer

        return SLIMTransformer
    raise AttributeError(name)
