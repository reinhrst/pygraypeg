Pure-Python Grayscale Image Encoder

port of https://github.com/Moodstocks/jpec

Use as you see fit, don't complain if it breaks (although push requests accepted)

Tested on python 3.4, use on other pythons at own risk

    encode(image)

expects `image` to be a numpy array of uint8s.

Quite honestly, any array will do, as long as it has a `.shape` property,
and can be addressed link `image[x, y]`. Array dimensions should be multiple
of 8 in both directions.

Result will be a bytestring, that should be a valid jpeg file
