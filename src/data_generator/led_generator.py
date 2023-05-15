from skmultiflow.data.led_generator import LEDGenerator
import click
import numpy as np

'''
Documentation
https://scikit-multiflow.readthedocs.io/en/stable/api/generated/skmultiflow.data.LEDGenerator.html#skmultiflow.data.LEDGenerator
'''


def concept(concept_size, excepting_number):
    stream = LEDGenerator(random_state=42, has_noise=False)

    data, label = stream.next_sample(concept_size)
    for num in excepting_number:
        data = data[label != num]
        # label = label[label != num]
    return data


@click.command()
@click.option('--concept_size', type=int)
@click.option('--num_concept', type=int)
def main(concept_size, n_concepts):
    concepts = []
    labels = []
    for n in range(n_concepts):
        c = concept(concept_size, [n % 10])
        concepts.append(c)
        labels += [1]
        labels += [0 for _ in range(concept_size - 1)]

    data_stream = np.concatenate(concepts, axis=0)
    _path = '../data/'
    _file_name = f'Realworld_LED_{concept_size}_{n_concepts}.txt'
    np.savetxt(_path + _file_name, data_stream, delimiter=", ")

    _label_file_name = f'Realworld_LED_{concept_size}_{n_concepts}_label.txt'
    np.savetxt(_path + _label_file_name, np.array(labels), delimiter=", ")


if __name__ == '__main__':
    main()
