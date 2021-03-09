#include <pthread.h>
#include <unistd.h>
#include <stdio.h>

static unsigned long long thread_nr = 0;

pthread_mutex_t mutex_;

void* inc_thread_nr(void* arg) {
    (void*)arg;
    pthread_mutex_lock(&mutex_);
    ++thread_nr;
    pthread_mutex_unlock(&mutex_);
    sleep(15);
	return NULL;
}   

#define MAXTHREADS 1000000
#define THREADSTACK  65536

int main(int argc, char *argv[])
{
    pthread_t pid[MAXTHREADS];
    pthread_attr_t attrs;
    int err, i;
    int cnt = 0;

    pthread_attr_init(&attrs);
    pthread_attr_setstacksize(&attrs, THREADSTACK);

    pthread_mutex_init(&mutex_, NULL);

    for (cnt = 0; cnt < MAXTHREADS; cnt++) {
        err = pthread_create(&pid[cnt], &attrs, (void*)inc_thread_nr, NULL);
        if (err != 0)
            break;
    }
    pthread_attr_destroy(&attrs);
    for (i = 0; i < cnt; i++)
        pthread_join(pid[i], NULL);
    pthread_mutex_destroy(&mutex_);
    printf("Maximum number of threads per process is %d (%llu)\n", cnt, thread_nr);
}