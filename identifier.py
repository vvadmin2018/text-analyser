class FuzzyDetective:
    """Нечёткий детектив - главный класс программы"""

    def __init__(self):
        self.authors = {}  # словарь {имя: AuthorProfile}

    def add_author(self, name, texts):
        """Добавляет автора с его текстами для обучения"""
        profile = AuthorProfile(name)
        profile.build_from_texts(texts)
        self.authors[name] = profile

    def identify(self, anonymous_text, threshold=0.5):
        """Определяет автора анонимного текста"""
        extractor = FeatureExtractor()
        features = extractor.extract(anonymous_text)

        results = {}
        for name, profile in self.authors.items():
            sim = profile.similarity(features)
            results[name] = sim

        # Находим лучшего
        best_author = max(results, key=results.get)
        best_score = results[best_author]

        if best_score < threshold:
            return "Автор не определён", results
        else:
            return best_author, results
