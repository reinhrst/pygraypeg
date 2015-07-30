from pygreypeg import encode


if __name__ == "__main__":
    import numpy
    import sys
    print("Running test mode, generating image and writing it to a file")
    print("Filename should be first argument")
    if len(sys.argv) != 2:
        print("""Run as "%s targetfile.jpg" """ % sys.argv[0])
        sys.exit(1)
    filename = sys.argv[1]
    image = numpy.zeros((256, 128), dtype="u1")
    for x in range(128):
        for y in range(256):
            image[y, x] = 255 * (y % 2)
    jpegdata = encode(image, 93)
    with open(filename, "wb") as f:
        f.write(jpegdata)
    print("test jpeg written to %s" % filename)
