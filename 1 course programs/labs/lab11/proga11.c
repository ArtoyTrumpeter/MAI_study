#include <stdio.h>
#define SPACE 0
#define SIGN 1
#define NUMBER 2
#define OTHER 3


int IS_DIGIT(char c)
{
    if ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'F')) {
        return 1;
    }
    else {
        return 0;
    }
}


int IS_SIGN(char c)
{
    if ((c == '+') || (c == '-')) {
        return 1;
    }
    else {
        return 0;
    }
}


int IS_SPACE(char c)
{
    if ((c == ' ') || (c == ',') || (c == '\t') || (c == '\n')) {
        return 1;
    }
    else {
        return 0;
    }
}


int main(void)
{
    int n = 0, count = 0;
    char c;
    int st = SPACE;
    do {
        c = getchar();
        switch (st) {
	    case SPACE:
                if (IS_DIGIT(c)) {
	            st = NUMBER;
                }
		else if (IS_SIGN(c)) {
	              st = SIGN;
	        }
		else if (IS_SPACE(c)) {
	              st = SPACE;
	        }
	        break;
	    case SIGN:
	        if (IS_SPACE(c)) {
	            st = SPACE;
		}
		else if (IS_DIGIT(c)) {
	            st = NUMBER;
		}
		else {
	            st = OTHER;
		}
	        break;
	    case NUMBER:
	        if ((n == 7) && (IS_SPACE(c))) {
	            st = SPACE;
	            ++count;
		    n = 0;
		}
		else if (IS_DIGIT(c)) {
	            st = NUMBER;
		    n++;
		}
		else if ((IS_SPACE(c)) && (n != 7)) {
		    st = SPACE;
		    n = 0;
		}
		else {
		    st = OTHER;
		}
		break;
	    case OTHER:
	        if (IS_SPACE(c)) {
	            st = SPACE;
		} 
	        break;
	}
    } while (c != EOF);
    printf ("%d\n", count);
    return 0;
}
