def get_unique_num(D, I, static_number):
    """ This function gets a parameter for number of unique components. Unique is a componenet with imag part of 0 or
        couple of conjugate couple """
    i = 0
    for j in range(static_number):
        index = len(I) - i - 1
        val = D[I[index]]

        if val.imag == 0:
            i = i + 1
        else:
            i = i + 2

    return i
