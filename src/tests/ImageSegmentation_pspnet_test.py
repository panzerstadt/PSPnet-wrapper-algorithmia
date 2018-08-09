from . import ImageSegmentation_pspnet

def test_ImageSegmentation_pspnet():
    assert ImageSegmentation_pspnet.apply("Jane") == "hello Jane"
