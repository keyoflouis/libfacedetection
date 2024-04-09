void cuda_dotProduct(const float* p1, const float* p2, float& sum, int num);

//void dotProductGPU(float& sum, const float* p1, const float* p2, const int num);


class DotProductGPU {
public:
    float* d_sum;
    float* d_vec1;
    float* d_vec2;
    const int num;
    const int blockSize = 256;
    int numBlocks;

    DotProductGPU(const int num_);

    void operator()(float& sum, const float* p1, const float* p2);

    ~DotProductGPU();
    
};