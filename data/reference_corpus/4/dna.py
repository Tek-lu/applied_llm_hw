import re


def dna(dna: str) -> str:


    if len(re.findall("[ATCG]", dna)) != len(dna):
        raise ValueError("Invalid Strand")

    return dna.translate(dna.maketrans("ATCG", "TAGC"))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
