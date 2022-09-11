import random
import re
import numpy as np
import argparse
from typing import Union
import pickle
import os
from dataclasses import dataclass, field


@dataclass
class NgrammModel:
    model_filename: str = field(default="model.pkl")
    model_weights: dict = field(default_factory=dict)
    model_n: int = field(default=4)

    def fit(self, filename: str, amount_of_gramms: int) -> tuple[bool, Union[Exception, None]]:
        try:
            with open(filename, 'r', encoding='UTF-8') as reading_file:
                text = reading_file.read()
        except FileNotFoundError as e:
            return False, e

        try:
            text = re.sub('[^а-яё]', ' ', text.lower())
            words = re.split('\\s+', text)
            n_grams = {}
            dictionary = {}
            for i in range(len(words) - (amount_of_gramms - 1)):
                next_words = []
                for k in range(amount_of_gramms - 1):
                    next_words.append(words[i + k + 1])
                if words[i] not in dictionary:
                    dictionary[words[i]] = [" ".join(next_words)]
                else:
                    dictionary[words[i]] += [" ".join(next_words)]

            for key in dictionary.keys():
                words = []
                for i in range(len(dictionary[key])):
                    words.append(dictionary[key][i])
                uniq_words = set(words)
                for uniq_word in uniq_words:
                    probability = words.count(uniq_word) / len(dictionary[key])
                    if key not in n_grams:
                        n_grams[key] = [(uniq_word, probability)]
                    else:
                        n_grams[key] += [(uniq_word, probability)]
            self.model_weights |= n_grams
            return True, None
        except Exception as e:
            return False, e

    def generate(self, length, amount_of_gramms=4, prefix=None) -> tuple[bool, Union[Exception, None]]:
        try:
            with open(self.model_filename, 'rb') as f:
                self.model_weights = pickle.load(f)
        except FileNotFoundError as e:
            return False, e
        self.model_n = amount_of_gramms
        try:
            if len(self.model_weights) != 0:
                if prefix:
                    prefix = ' '.join(prefix)
                else:
                    prefix = random.choice(list(self.model_weights.keys()))
                gen_text = [prefix]
                if prefix.split()[-1] not in self.model_weights:
                    prefix = random.choice(list(self.model_weights.keys()))
                else:
                    prefix = str(prefix.split()[-1])
                amount_of_words = len(gen_text[0].split())
                while amount_of_words < length:
                    var_list = []
                    word_list = []
                    for i in range(len(self.model_weights[prefix])):  # составляем список весов
                        var_list.append(self.model_weights[prefix][i][1])
                    for i in range(len(self.model_weights[prefix])):  # составляем список вариантов следущих слов
                        word_list.append(self.model_weights[prefix][i][0])
                    next_words = np.random.choice(word_list, 1, p=var_list)[0]
                    if length - 1 - amount_of_words > self.model_n - 1:  # проверяем, чтобы количество слов в
                        # сгенерированном тексте не превышало заданное
                        gen_text.append(next_words)
                    else:
                        next_words = next_words.split()
                        gen_text.append(' '.join(next_words[:length - amount_of_words]))  # отрезаем лишние слова
                        next_words = " ".join(next_words)
                    if next_words.split()[-1] in self.model_weights:  # проверяем, есть ли последнее слово в ключах
                        prefix = next_words.split()[-1]
                    else:
                        prefix = random.choice(list(self.model_weights.keys()))  # если такого слова ключах нет,
                        # выбираем рандомно
                    amount_of_words += self.model_n - 1
                generated_text = ' '.join(gen_text)
            else:
                generated_text = "Текст слишком маленький для построение такой модели"

            yield generated_text
            with open('generated.txt', 'w') as f:
                f.write(generated_text)
            return True, str

        except Exception as e:
            return False, e

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with open(file=self.model_filename, mode='wb') as file:
            pickle.dump(self.model_weights, file)


def main():
    parser = argparse.ArgumentParser(description='ngramm-Model')
    parser.add_argument("--input-dir", help="Input directory")
    parser.add_argument("--model", help="Model filename")
    parser.add_argument("--n", nargs='?', help="Choose amount of gramms in ngramm-model", default=4)
    args = parser.parse_args()
    with NgrammModel(model_filename=args.model) as model:
        for filename in os.listdir(args.input_dir):
            processed, err = model.fit(args.input_dir + "/" + filename, int(args.n))
            if processed:
                print(filename, "processed")
            else:
                print(err)


if __name__ == '__main__':
    main()
