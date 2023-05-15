import os.path
import warnings

import numpy as np
import random

seed = 42
random.seed(seed)



def _ADD_function(x1, x2, c):
    cor_feature = [x1 + x2]

    return cor_feature


def _MUL_function(x1, x2, c):
    cor_feature = [x1 * x2]

    return cor_feature


def _FACTOR_function(x1, x2, c):
    c1 = 2
    c2 = 3
    cor_feature = [x1 * c1 + x2 * c2]

    return cor_feature


def _AND_function(x1, x2, c):
    cor_feature = [np.array([1 if (x1[i] > c / 2) and (x2[i] > c / 2) else 0 for i in range(len(x1))])]

    return cor_feature


def _OR_function(x1, x2, c):
    cor_feature = [np.array([1 if (x1[i] > c / 2) or (x2[i] > c / 2) else 0 for i in range(len(x1))])]

    return cor_feature


def _XOR_function(x1, x2, c):
    cor_feature = [np.array([1 if (x1[i] > c / 2) != (x2[i] > c / 2) else 0 for i in range(len(x1))])]

    return cor_feature


def _MIX_function(concept_size, c, seed):
    random.seed(seed-1)
    dim1 = np.array([random.randint(1, c) for _ in range(concept_size)])
    random.seed(seed-2)
    dim2 = np.array([random.randint(1, c) for _ in range(concept_size)])
    random.seed(seed-3)
    dim3 = np.array([random.randint(1, c) for _ in range(concept_size)])
    X = [dim1, dim2, dim3]

    num_features_additional = 7
    functions = ["ADD", "MUL", "FACTOR", "AND", "OR", "XOR"]

    for j in range(num_features_additional):
        random.seed(seed)
        rand1 = random.randint(0, 2)
        random.seed(seed+1)
        rand2 = random.randint(0, 2)
        random.seed(seed+2)
        i = random.randint(0, len(functions) - 1)
        new_feature = eval(f'_{functions[i]}_function')(X[rand1], X[rand2], c)
        X += new_feature

    return X


def _generate_concept(x1, x2, correlation_function, categories_per_feature):
    features = []

    cor_feature = eval(f'_{correlation_function}_function')(x1, x2, categories_per_feature) if correlation_function != 'MIX' else _MIX_function(x1.shape[0], categories_per_feature, seed)
    features += cor_feature

    return features


def _generate_drift(num_concepts,
                    correlation_function,
                    concept_size,
                    categories_per_feature):
    data_stream = []
    for n in range(num_concepts):
        random.seed(seed)
        dim1 = np.array([random.randint(1, categories_per_feature) for _ in range(concept_size)])
        random.seed(seed + 1)
        dim2 = np.array([random.randint(1, categories_per_feature) for _ in range(concept_size)])
        random.seed(seed + 2)
        dim3 = np.array([random.randint(1, categories_per_feature) for _ in range(concept_size)])
        random.seed(seed + 3)
        dim_noise = np.array(
            [random.randint(10 * categories_per_feature + 1, 11 * categories_per_feature) for _ in range(concept_size)])

        features = [dim1, dim2, dim3]
        new_features = _generate_concept(eval('dim' + str(1 + n % 3)),
                                         eval('dim' + str(1 + (n + 1) % 3)),
                                         correlation_function,
                                         categories_per_feature)
        features += [f for f in new_features]
        features += [dim_noise]
        concept = np.concatenate([x.reshape(-1, 1) for x in features], axis=1)
        data_stream.append(concept)
    return np.concatenate(data_stream, axis=0)


def _generate_label(concept_size, num_concepts):
    label = []
    for _ in range(num_concepts):
        label += [1]
        label += [0 for _ in range(concept_size - 1)]
    label[0] = 0

    return label


def run_generation(data_folder, concept_size, categories_per_feature, num_concepts, correlation_func):
    _file_name = f'Synthetic_{num_concepts}Concepts_{correlation_func}_{concept_size}dataPerConcept_{categories_per_feature}category'
    _label_name = f'{_file_name}_label.txt'

    if os.path.exists(os.path.join(data_folder, _file_name)):
        warnings.warn(f'File already exist: {_file_name}')
        return

    data_stream = _generate_drift(num_concepts,
                                  correlation_func,
                                  concept_size=concept_size,
                                  categories_per_feature=categories_per_feature)

    data_stream_label = _generate_label(concept_size, num_concepts)

    np.savetxt(os.path.join(data_folder, _file_name + '.txt'), data_stream, delimiter=", ")
    np.savetxt(os.path.join(data_folder, _label_name), np.array(data_stream_label), delimiter=", ")
