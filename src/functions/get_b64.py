from src.python_utils.utils import image_to_base64, get_filepath

dummy_file = get_filepath('dummy/01.jpg')
dummy_output = 'dummy/01_result.jpg'

def get_dummy_base64():
    return image_to_base64(dummy_file)


if __name__ == '__main__':
    dummy_file = '../dummy/01.jpg'
    dummy_output = '../dummy/01_result.jpg'
    t = get_dummy_base64()

    print(t)

    with open('temp.txt', 'w') as f:
        f.write(str(t))