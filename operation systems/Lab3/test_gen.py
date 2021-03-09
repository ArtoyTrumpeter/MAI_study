#!/usr/bin/env python3
import random
import sys


def matrix_gen(size):
	for _ in range(size):
		for _ in range(size):
			print(random.randint(-100, 100), end=" ")
		print()


def test_gen(size):
	print(random.randint(1, 2))
	print(random.randint(3, 5), size)
	matrix_gen(size)


def main():
	size = int(sys.argv[1])
	test_gen(size)


if __name__ == "__main__":
	main()