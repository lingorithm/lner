
import sys
from .ner import NER


def lner(lang, sentences, entities, **kwargs):
    """[summary]

    Args:
        lang (str): ISO language naming
        processors (list): List for processing processors
        options (dict, optional): options used by the processores. Defaults to {}.

    Returns:
        class: class corosponding to the language
    """
    return NER(lang, sentences, entities, **kwargs)


sys.modules[__name__] = lner
