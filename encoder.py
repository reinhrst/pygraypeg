import constants
import struct


class _Huffman(object):
    def __init__(self):
        self._dc = 0
        self._buf = bytearray()
        self._bits = 0
        self._nbits = 0

    def write_bits(self, bits, nbits):
        mask = (1 << nbits) - 1
        self._bits <<= nbits
        self._bits |= bits & mask
        self._nbits += nbits
        while self._nbits >= 8:
            val = self._bits >> (self._nbits - 8)
            assert val < 256 and val >= 0, \
                "value = %x (%x, %d)" % (val, self._bits, self._nbits)
            self._buf.append(val)
            if val == 0xFF:
                # special rule because 0xFF is special
                self._buf.append(0)
            self._nbits -= 8
            mask = (1 << self._nbits) - 1
            self._bits &= mask

    @staticmethod
    def position_of_highest_1bit(value):
        count = 0
        while value:
            count += 1
            value >>= 1
        return count

    def encode_block(self, zz, length):
        if length > 0:
            val = zz[0] - self._dc
            self._dc = zz[0]
        else:
            val = -self._dc
            self._dc = 0

        bits = val
        if val < 0:
            val = -val
            bits = ~val
        nbits = _Huffman.position_of_highest_1bit(val)
        self.write_bits(constants.dc_code[nbits], constants.dc_len[nbits])
        if nbits:
            self.write_bits(bits, nbits)

        # AC coefficients encoding (w/ RLE of zeroes)
        nz = 0
        for i in range(1, length):
            val = zz[i]
            if val == 0:
                nz += 1
            else:
                while nz >= 16:
                    # ZRL code
                    self.write_bits(
                        constants.ac_code[0xF0], constants.ac_len[0xF0])
                    nz -= 16
                bits = val
                if val < 0:
                    val = -val
                    bits = ~val
                nbits = _Huffman.position_of_highest_1bit(val)
                j = (nz << 4) + nbits
                self.write_bits(constants.ac_code[j], constants.ac_len[j])
                if nbits:
                    self.write_bits(bits, nbits)
                nz = 0
        if length < 64:
            # EOB marker
            self.write_bits(constants.ac_code[0x00], constants.ac_len[0x00])

    def end_and_get_buffer(self):
        # todo: what to do with bits that haven't made it to _buf yet?
        return self._buf


def _dct_matrix_for_quality(quality):
    """
    create DCT matrix based on quality
    """
    if (quality < 50):
        scale = 50.0 / quality
    else:
        scale = 2 - quality / 50.0
    dqt = [min(255, max(0, int(val * scale + 0.5))) for val in constants.qzr]
    assert(len(dqt) == 64)
    return dqt


def _get_header(image, dqt):
    """
    returns bytearray with the header
    """
    buf = bytearray()

    def writebyte(val):
        buf.extend(struct.pack(">B", val))

    def writeshort(val):
        buf.extend(struct.pack(">H", val))

    width, height = image.shape
    # SOI
    writeshort(0xFFD8)  # SOI marker

    # APP0
    writeshort(0xFFE0)  # APP0 marker
    writeshort(0x0010)  # segment length
    writebyte(0x4A)     # 'J'
    writebyte(0x46)     # 'F'
    writebyte(0x49)     # 'I'
    writebyte(0x46)     # 'F'
    writebyte(0x00)     # '\0'
    writeshort(0x0101)  # v1.1
    writebyte(0x00)     # no density unit
    writeshort(0x0001)  # X density = 1
    writeshort(0x0001)  # Y density = 1
    writebyte(0x00)     # thumbnail width = 0
    writebyte(0x00)     # thumbnail height = 0

    # DQT
    writeshort(0xFFDB)  # DQT marker
    writeshort(0x0043)  # segment length
    writebyte(0x00)     # table 0, 8-bit precision (0)
    for index in constants.zz:
        writebyte(dqt[index])

    # SOF0
    writeshort(0xFFC0)  # SOF0 marker
    writeshort(0x000B)  # segment length
    writebyte(0x08)     # 8-bit precision
    writeshort(height)
    writeshort(width)
    writebyte(0x01)     # 1 component only (grayscale)
    writebyte(0x01)     # component ID = 1
    writebyte(0x11)     # no subsampling
    writebyte(0x00)     # quantization table 0

    # DHT
    writeshort(0xFFC4)                     # DHT marker
    writeshort(19 + constants.dc_nb_vals)  # segment length
    writebyte(0x00)                      # table 0 (DC), type 0 (0 = Y, 1 = UV)
    for node in constants.dc_nodes[1:]:
        writebyte(node)
    for val in constants.dc_vals:
        writebyte(val)

    writeshort(0xFFC4)                     # DHT marker
    writeshort(19 + constants.ac_nb_vals)
    writebyte(0x10)                      # table 1 (AC), type 0 (0 = Y, 1 = UV)
    for node in constants.ac_nodes[1:]:
        writebyte(node)
    for val in constants.ac_vals:
        writebyte(val)

    # SOS
    writeshort(0xFFDA)  # SOS marker
    writeshort(8)       # segment length
    writebyte(0x01)     # nb. components
    writebyte(0x01)     # Y component ID
    writebyte(0x00)     # Y HT = 0
    # segment end
    writebyte(0x00)
    writebyte(0x3F)
    writebyte(0x00)

    return buf


def _block_iterator(image):
    width, height = image.shape
    x = y = 0
    while y != height:
        assert y < height
        yield image[x:x+8, y:y+8]
        x += 8
        if x == width:
            x = 0
            y += 8


def _block_dct(block):
    """
    calculates and returns the block dct
    """
    coeff = constants.dct
    tmp = [None] * 64
    for y, row in enumerate(block):
        s = [float(i) + j - 256 for i, j in zip(row[0:4], row[4:8][::-1])]
        d = [float(i) - j for i, j in zip(row[0:4], row[4:8][::-1])]

        tmp[8*y] = coeff[3]*(s[0]+s[1]+s[2]+s[3])
        tmp[8*y+1] = coeff[0]*d[0]+coeff[2]*d[1]+coeff[4]*d[2]+coeff[6]*d[3]
        tmp[8*y+2] = coeff[1]*(s[0]-s[3])+coeff[5]*(s[1]-s[2])
        tmp[8*y+3] = coeff[2]*d[0]-coeff[6]*d[1]-coeff[0]*d[2]-coeff[4]*d[3]
        tmp[8*y+4] = coeff[3]*(s[0]-s[1]-s[2]+s[3])
        tmp[8*y+5] = coeff[4]*d[0]-coeff[0]*d[1]+coeff[6]*d[2]+coeff[2]*d[3]
        tmp[8*y+6] = coeff[5]*(s[0]-s[3])+coeff[1]*(s[2]-s[1])
        tmp[8*y+7] = coeff[6]*d[0]-coeff[4]*d[1]+coeff[2]*d[2]-coeff[0]*d[3]

    dct = [None] * 64
    for x, col in enumerate(block.transpose()):
        s = [tmp[x + i] + tmp[x + 56 - i] for i in range(0, 32, 8)]
        d = [tmp[x + i] - tmp[x + 56 - i] for i in range(0, 32, 8)]

        dct[x] = coeff[3]*(s[0]+s[1]+s[2]+s[3])
        dct[8+x] = coeff[0]*d[0]+coeff[2]*d[1]+coeff[4]*d[2]+coeff[6]*d[3]
        dct[16+x] = coeff[1]*(s[0]-s[3])+coeff[5]*(s[1]-s[2])
        dct[24+x] = coeff[2]*d[0]-coeff[6]*d[1]-coeff[0]*d[2]-coeff[4]*d[3]
        dct[32+x] = coeff[3]*(s[0]-s[1]-s[2]+s[3])
        dct[40+x] = coeff[4]*d[0]-coeff[0]*d[1]+coeff[6]*d[2]+coeff[2]*d[3]
        dct[48+x] = coeff[5]*(s[0]-s[3])+coeff[1]*(s[2]-s[1])
        dct[56+x] = coeff[6]*d[0]-coeff[4]*d[1]+coeff[2]*d[2]-coeff[0]*d[3]
    return dct


def _block_quant(dct, dqt):
    return [dct[i] / dqt[i] for i in range(64)]


def _block_zz(quant):
    zz = [int(quant[zz_index]) for zz_index in constants.zz]
    length = len([x for x in zz if x])
    return zz, length


def encode(image, quality):
    width, height = image.shape
    assert width % 8 == 0, "width should be multiple of 8"
    assert height % 8 == 0, "height should be multiple of 8"

    buf = bytearray()
    dqt = _dct_matrix_for_quality(quality)
    huf = _Huffman()

    buf.extend(_get_header(image, dqt))

    for block in _block_iterator(image):
        dct = _block_dct(block)
        quant = _block_quant(dct, dqt)
        zz, length = _block_zz(quant)

        # huffman encoding
        huf.encode_block(zz, length)

    buf.extend(huf.end_and_get_buffer())
    buf.extend(struct.pack(">H", 0xFFD9))  # EOI marker

    return buf

if __name__ == "__main__":
    import numpy
    import sys
    import os
    print("Running test mode, generating image and writing it to a file")
    print("Filename should be first argument")
    if len(sys.argv) != 2:
        print("""Run as "%s targetfile.jpg" """ % sys.argv[0])
        sys.exit(1)
    filename = sys.argv[1]
    if os.path.exists(filename):
        print("File %s exists, not running", filename)
        sys.exit(1)
    image = numpy.zeros((128, 128), dtype="u1")
    for x in range(128):
        for y in range(128):
            image[x, y] = x + y
    jpegdata = encode(image, 93)
    with open(filename, "wb") as f:
        f.write(jpegdata)
    print("test jpeg written to %s" % filename)
