Pure-Python Grayscale Image Encoder

inspiration drawn from  https://github.com/Moodstocks/jpec

Use as you see fit, don't complain if it breaks (although push requests accepted)

Tested on python 3.4, use on other pythons at own risk

    encode(image, quality)

expects `image` to be a numpy array of uint8s. Quality is a number between
0 (low quality) and 100 (high quality).

Quite honestly, any array will do for image, as long as it has a `.shape` property,
and can be addressed link `image[x, y]`. Array dimensions should be multiple
of 8 in both directions.

Result will be a bytestring, that should be a valid jpeg file
