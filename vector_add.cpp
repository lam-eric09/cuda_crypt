#include <cstdlib>
#define N 200000000 

void vector_add(float *out, float *a, float *b, int n){
    for(int i=0; i<n; i++){
        out[i] = a[i] + b[i];
    }
}

int run_one_iter(){
    float *out, *a, *b;
    a = (float*)malloc(sizeof(float)*N);
    b = (float*)malloc(sizeof(float)*N);
    out = (float*)malloc(sizeof(float)*N);

    for(int i=0; i<N; i++){
        a[i] = i * 1.0f;
        b[i] = i * 2.0f;
    }
    vector_add(out, a, b, N);

    free(a);
    free(b);
    free(out);
    return 0;
}

int main(){
    int n = 10;
    for(int i=0; i<n; i++){
        run_one_iter();
    }
    return 0;
}