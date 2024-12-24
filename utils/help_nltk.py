import os
import nltk
from typing import Dict, List, Optional

resource_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")

class NLTKHelper:
    _stopwords: Dict[str, Optional[List[str]]] = {
        "english": None,
        "chinese": None,
    }

    def get_stopwords(self, lang: str) -> List[str]:
        nltk.data.path.append(os.path.join(resource_path, "nltk_data"))
        if self._stopwords[lang] is None:
            try:
                nltk.data.find("corpora/stopwords")
            except LookupError:
                nltk.download("stopwords", download_dir=os.path.join(resource_path, "nltk_data"))

            self._stopwords[lang] = nltk.corpus.stopwords.words(lang)
        return self._stopwords[lang]
