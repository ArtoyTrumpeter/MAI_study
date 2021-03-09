#include <stdio.h>

int code(char c){
  int n = 0;
  while(c != ' '){
    if((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')){
      n++;
      c += n + 3;
      if(c > 'z'){
	  c -= 26;
      }
      putchar(c);
    }
  }
  if(c = ' '){
    n = 1;
  }
  return 0;
}

int main(){
  char c = 0;
  while(c != EOF){
    code(c = getchar());
  }
  return 0;
}
