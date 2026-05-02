import numpy as np
from feature_extractor import FeatureExtractor
import config


class TriangularMembership:
    """Треугольная функция принадлежности (смягченная)"""

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        # Добавляем "размытие" границ
        softening = config.SOFTENING
        self.softening = softening * (c - a)

    def mu(self, x):
        # Расширяем границы на величину softening
        a_soft = self.a - self.softening
        c_soft = self.c + self.softening

        if x <= a_soft or x >= c_soft:
            return 0.001
        elif x <= self.b:
            if self.b == self.a:
                return 1.0
            # Плавный подъем от a_soft до b
            return (x - a_soft) / (self.b - a_soft)
        else:
            if self.c == self.b:
                return 1.0
            # Плавный спуск от b до c_soft
            return (c_soft - x) / (c_soft - self.b)


class AuthorProfile:
    """Нечёткий портрет автора"""

    def __init__(self, name):
        self.name = name
        self.features = []  # список функций принадлежности
        self.feature_names = config.FEATURE_LIST

    def build_from_texts(self, texts):
        """
        texts: список текстов этого автора (каждый текст -> строка)
        """
        import numpy as np
        from feature_extractor import FeatureExtractor

        num_props = config.N_FEATURES

        print(f"\n🖌️  Строим портрет для {self.name}")

        # Собираем все значения по каждому признаку
        all_values = [[] for _ in range(num_props)]

        extractor = FeatureExtractor()

        for text_idx, text in enumerate(texts):
            print(f"  Обработка текста {text_idx + 1}/{len(texts)}...")

            # проверяем тип text
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='ignore')
            elif not isinstance(text, str):
                text = str(text)

            try:
                features = extractor.extract(text)
                print(f"    Извлечено признаков: {len(features)}")

                # Проверяем, что признаки извлечены правильно
                if len(features) == num_props:
                    for i, val in enumerate(features):
                        all_values[i].append(val)
                    print(f"    Значения: {[f'{v:.3f}' for v in features[:5]]}...")
                else:
                    print(f"    ❌ Ожидалось {num_props} признаков, получено {len(features)}")

            except Exception as e:
                print(f"  ❌ Ошибка при извлечении признаков: {e}")

        # Проверяем, что собрали данные
        print(f"\n  Статистика собранных данных:")
        for i in range(num_props):
            print(f"    Признак {i}: {len(all_values[i])} значений")
            print(all_values[i])
            if all_values[i]:
                print(f"      min={min(all_values[i]):.3f}, max={max(all_values[i]):.3f}")

        # Строим треугольные функции
        self.features = []
        for i in range(num_props):
            values = all_values[i]

            a = min(values)
            c = max(values)
            #b = np.median(values)
            b = (a+c)/2

            # Защита от a == b == c
            if b == c or a == b:
                a = 0.5 * a
                c = 1.5 * c
                b = (a+c)/2
                print(f"СРАБОТКА a == b == c,a= {a} ,b= {b},c= {c}")

            # Создаем функцию принадлежности
            self.features.append(TriangularMembership(a, b, c))
            if config.LEVEL_LOG == "DEBUG":
                print(f"    Признак {i}: a={a:.3f}, b={b:.3f}, c={c:.3f}")

        print(f"  ✅ Портрет для {self.name} построен! Всего функций: {len(self.features)}")
        return self.features

    def get_weights(self):
        """
        Возвращает веса признаков по умолчанию
        """
        # Веса по умолчанию для признаков (можно настроить)
        default_weights = [
            1.1,  # 0: Предл (средняя длина предложения)
            1.2,  # 1: Дисп предл (разброс от средней величины)
            1.1,  # 2: Абзац (длина абзаца)
            0.9,  # 3: ? (вопросительные)
            0.9,  # 4: ! (восклицательные)
            0.9,  # 5: ... (троеточия)
            1.0,  # 6: прямая речь
            1.1,  # 7: TTR (лексическое богатство)
            1.1,  # 8: Сущ (существительные)
            1.1,  # 9: Глаг (глаголы)
            1.1,  # 10: Прил (прилагательные)
            1.1,  # 11: ДлСл (длина слова)
            1.1,  # 12: , (запятые)
            1.05,  # 13: — (тире)
            0.9,  # 14: : (двоеточия)
            1.0,  # 15: Союз (союзы)
            1.0,  # 16: Предлоги (предлоги)
        ]

        # Убеждаемся, что весов столько же, сколько признаков
        while len(default_weights) < len(self.features):
            default_weights.append(1.0)

        # Обрезаем, если весов слишком много
        return default_weights[:len(self.features)]

    def similarity(self, text_features, use_weights=True, custom_weights=None):
        """
        Вычисляет сходство с вектором признаков текста
        """
        import numpy as np

        num_props = config.N_FEATURES

        # Проверяем, что есть признаки для сравнения
        if not self.features:
            print(f"  ⚠️ У автора {self.name} нет признаков для сравнения!")
            return 0.0

        # Вычисляем степень принадлежности для каждого признака
        similarities = []
        for i, feat_func in enumerate(self.features):
            if i < len(text_features):
                mu = feat_func.mu(text_features[i])
                similarities.append(mu)
            else:
                similarities.append(0.0)

        if not use_weights:
            return np.mean(similarities)

        # Получаем веса
        if custom_weights is not None:
            weights = custom_weights
        else:
            weights = self.get_weights()

        # Убеждаемся, что у нас достаточно весов
        while len(weights) < len(similarities):
            weights.append(1.0)
        weights = weights[:len(similarities)]

        # Вычисляем взвешенное среднее
        weighted_sum = 0
        total_weight = 0

        if config.LEVEL_LOG == "DEBUG":
            print(f"\n   Взвешенное сходство для {self.name}:")
            print(f"    {'№':<2} {'μ':<6} {'вес':<5} {'вклад':<8}")
            print(f"    {'-' * 25}")

        for i, (mu, weight) in enumerate(zip(similarities, weights)):
            contribution = mu * weight
            weighted_sum += contribution
            total_weight += weight

            if i < num_props:
                print(f"    {i:<2} {mu:<6.3f} {weight:<5.1f} {contribution:<8.3f}")

        result = weighted_sum / total_weight if total_weight > 0 else 0
        print(f"    {'-' * 25}")
        print(f"    ИТОГО: {result:.3f}")

        return result

    def similarity_with_details(self, text_features):
        """
        Возвращает сходство и детали (μ, веса, вклады)
        """

        # Проверяем, что есть признаки для сравнения
        if not self.features:
            print(f"  ⚠️ У автора {self.name} нет признаков для сравнения!")
            return 0.0, ([], [], [])

        # Вычисляем степень принадлежности
        similarities = []
        for i, feat_func in enumerate(self.features):
            if i < len(text_features):
                mu = feat_func.mu(text_features[i])
                similarities.append(mu)
                if config.LEVEL_LOG == "DEBUG":
                    print(f"  ⚠️ У автора признак {i}, степень принадлежности {feat_func.mu(text_features[i])}")
            else:
                similarities.append(0.0)

        # Получаем веса
        weights = self.get_weights()

        # Вычисляем вклады
        contributions = [s * w for s, w in zip(similarities, weights)]

        # Вычисляем итоговое сходство (взвешенное среднее)
        total = sum(contributions) / sum(weights) if sum(weights) > 0 else 0

        return total, (similarities, weights, contributions)

    def experiment_with_weights(self, text_features, weight_sets):
        """
        Позволяет экспериментировать с разными наборами весов
        """
        results = {}

        # Базовый результат без весов
        base_result = self.similarity(text_features, use_weights=False)
        results['Без весов'] = base_result

        # Результаты с разными весами
        for exp_name, weights in weight_sets.items():
            result = self.similarity(text_features, use_weights=True, custom_weights=weights)
            results[exp_name] = result

        return results