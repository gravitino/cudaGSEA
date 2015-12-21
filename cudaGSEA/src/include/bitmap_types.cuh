#ifndef CUDA_GSEA_BITMAP_TYPES_CUH
#define CUDA_GSEA_BITMAP_TYPES_CUH

#include <string>

struct bitmap8_t {

    unsigned char x;

    template <class index_t> __device__ __host__ __forceinline__
    bool getbit (index_t id) const {
        return (x >> id) & 1;
    }

    template <class index_t> __device__ __host__ __forceinline__
    void setbit (index_t id) {
        x |= 1U << id;
    }

    template <class index_t> __device__ __host__ __forceinline__
    void delbit (index_t id) {
        x &= ~(1U << id);
    }

    template <class index_t> __device__ __host__ __forceinline__
    void flipbit (index_t id) {
        x ^= ~(1U << id);
    }

    __device__ __host__ __forceinline__
    void zero () { x = 0; }

    __host__
    std::string to_string () const {
        std::string result = "";
        for (unsigned int id = 0; id < 8; id++)
            result = std::to_string(getbit(id))+result;
        return result;
    }
};

struct bitmap16_t {

    unsigned short x;

    template <class index_t> __device__ __host__ __forceinline__
    bool getbit (index_t id) const {
        return (x >> id) & 1;
    }

    template <class index_t> __device__ __host__ __forceinline__
    void setbit (index_t id) {
        x |= 1U << id;
    }

    template <class index_t> __device__ __host__ __forceinline__
    void delbit (index_t id) {
        x &= ~(1U << id);
    }

    template <class index_t> __device__ __host__ __forceinline__
    void flipbit (index_t id) {
        x ^= ~(1U << id);
    }

    __device__ __host__ __forceinline__
    void zero () { x = 0; }

    __host__
    std::string to_string () const {
        std::string result = "";
        for (unsigned int id = 0; id < 16; id++)
            result = std::to_string(getbit(id))+result;
        return result;
    }
};

struct bitmap32_t {

    unsigned int x;

    template <class index_t> __device__ __host__ __forceinline__
    bool getbit (index_t id) const {
        return (x >> id) & 1;
    }

    template <class index_t> __device__ __host__ __forceinline__
    void setbit (index_t id) {
        x |= 1U << id;
    }

    template <class index_t> __device__ __host__ __forceinline__
    void delbit (index_t id) {
        x &= ~(1U << id);
    }

    template <class index_t> __device__ __host__ __forceinline__
    void flipbit (index_t id) {
        x ^= ~(1U << id);
    }

    __device__ __host__ __forceinline__
    void zero () { x = 0; }

    __host__
    std::string to_string () const {
        std::string result = "";
        for (unsigned int id = 0; id < 32; id++)
            result = std::to_string(getbit(id))+result;
        return result;
    }
};

struct bitmap64_t {

    unsigned long int x;

    template <class index_t> __device__ __host__ __forceinline__
    bool getbit (index_t id) const {
        return (x >> id) & 1;
    }

    template <class index_t> __device__ __host__ __forceinline__
    void setbit (index_t id) {
        x |= 1UL << id;
    }

    template <class index_t> __device__ __host__ __forceinline__
    void delbit (index_t id) {
        x &= ~(1UL << id);
    }

    template <class index_t> __device__ __host__ __forceinline__
    void flipbit (index_t id) {
        x ^= ~(1UL << id);
    }

    __device__ __host__ __forceinline__
    void zero () { x = 0; }

    __host__
    std::string to_string () const {
        std::string result = "";
        for (unsigned int id = 0; id < 64; id++)
            result = std::to_string(getbit(id))+result;
        return result;
    }
};

struct bitmap128_t {

    unsigned long int x; // bits 64..127  (slot == 1)
    unsigned long int y; // bits  0.. 63  (slot == 0)
    
    template <class index_t> __device__ __host__ __forceinline__
    bool getbit (index_t id) const {
        const index_t slot = id >> 6;
        const unsigned long int& current = slot ? x : y;
        return (current >> (id-(slot << 6))) & 1;
    }

    template <class index_t> __device__ __host__ __forceinline__
    void setbit (index_t id) {
        const index_t slot = id >> 6;
        unsigned long int& current = slot ? x : y;
        current |= 1UL << (id-(slot << 6));
    }

    template <class index_t> __device__ __host__ __forceinline__
    void delbit (index_t id) {
        const index_t slot = id >> 6;
        unsigned long int& current = slot ? x : y;
        current &= ~(1UL << (id-(slot << 6)));
    }

    template <class index_t> __device__ __host__ __forceinline__
    void flipbit (index_t id) {
        const index_t slot = id >> 6;
        unsigned long int& current = slot ? x : y;
        current ^= 1UL << (id-(slot << 6));
    }

    __device__ __host__ __forceinline__
    void zero () { x = 0; y = 0; }

    __host__
    std::string to_string () const {
        std::string result = "";
        for (unsigned int id = 0; id < 128; id++)
            result = std::to_string(getbit(id))+result;
        return result;
    }
};

struct bitmap256_t {

    unsigned long int x; // bits 192..255   (slot == 3)
    unsigned long int y; // bits 128..191   (slot == 2)
    unsigned long int z; // bits  64..127   (slot == 1)
    unsigned long int w; // bits   0.. 63   (slot == 0)
    
    template <class index_t> __device__ __host__ __forceinline__
    bool getbit (index_t id) const {
        const index_t slot = id >> 6;
        const unsigned long int& current = slot == 3 ? x :
                                           slot == 2 ? y :
                                           slot == 1 ? z : w;
        return (current >> (id-(slot << 6))) & 1;
    }

    template <class index_t> __device__ __host__ __forceinline__
    void setbit (index_t id) {
        const index_t slot = id >> 6;
        unsigned long int& current = slot == 3 ? x :
                                     slot == 2 ? y :
                                     slot == 1 ? z : w;
        current |= 1UL << (id-(slot << 6));
    }

    template <class index_t> __device__ __host__ __forceinline__
    void delbit (index_t id) {
        const index_t slot = id >> 6;
        unsigned long int& current = slot == 3 ? x :
                                     slot == 2 ? y :
                                     slot == 1 ? z : w;
        current &= ~(1UL << (id-(slot << 6)));
    }

    template <class index_t> __device__ __host__ __forceinline__
    void flipbit (index_t id) {
        const index_t slot = id >> 6;
        unsigned long int& current = slot == 3 ? x :
                                     slot == 2 ? y :
                                     slot == 1 ? z : w;
        current ^= 1UL << (id-(slot << 6));
    }

    __device__ __host__ __forceinline__
    void zero () { x = 0; y = 0; z = 0; w = 0;}

    __host__
    std::string to_string () const {
        std::string result = "";
        for (unsigned int id = 0; id < 256; id++)
            result = std::to_string(getbit(id))+result;
        return result;
    }
};

struct bitmap512_t {

    unsigned long int X; // bits 448..511   (slot == 7)
    unsigned long int Y; // bits 384..447   (slot == 6)
    unsigned long int Z; // bits 320..383   (slot == 5)
    unsigned long int W; // bits 256..319   (slot == 4)
    unsigned long int x; // bits 192..255   (slot == 3)
    unsigned long int y; // bits 128..191   (slot == 2)
    unsigned long int z; // bits  64..127   (slot == 1)
    unsigned long int w; // bits   0.. 63   (slot == 0)
    
    template <class index_t> __device__ __host__ __forceinline__
    bool getbit (index_t id) const {
        const index_t slot = id >> 6;
        const unsigned long int& current = slot == 7 ? X :
                                           slot == 6 ? Y :
                                           slot == 5 ? Z :
                                           slot == 4 ? W :
                                           slot == 3 ? x :
                                           slot == 2 ? y :
                                           slot == 1 ? z : w;
        return (current >> (id-(slot << 6))) & 1;
    }

    template <class index_t> __device__ __host__ __forceinline__
    void setbit (index_t id) {
        const index_t slot = id >> 6;
        unsigned long int& current = slot == 7 ? X :
                                     slot == 6 ? Y :
                                     slot == 5 ? Z :
                                     slot == 4 ? W :
                                     slot == 3 ? x :
                                     slot == 2 ? y :
                                     slot == 1 ? z : w;
        current |= 1UL << (id-(slot << 6));
    }

    template <class index_t> __device__ __host__ __forceinline__
    void delbit (index_t id) {
        const index_t slot = id >> 6;
        unsigned long int& current = slot == 7 ? X :
                                     slot == 6 ? Y :
                                     slot == 5 ? Z :
                                     slot == 4 ? W :
                                     slot == 3 ? x :
                                     slot == 2 ? y :
                                     slot == 1 ? z : w;
        current &= ~(1UL << (id-(slot << 6)));
    }

    template <class index_t> __device__ __host__ __forceinline__
    void flipbit (index_t id) {
        const index_t slot = id >> 6;
        unsigned long int& current = slot == 7 ? X :
                                     slot == 6 ? Y :
                                     slot == 5 ? Z :
                                     slot == 4 ? W :
                                     slot == 3 ? x :
                                     slot == 2 ? y :
                                     slot == 1 ? z : w;
        current ^= 1UL << (id-(slot << 6));
    }

    __device__ __host__ __forceinline__
    void zero () {X = 0; Y = 0; Z = 0; W = 0; x = 0; y = 0; z = 0; w = 0;}

    __host__
    std::string to_string () const {
        std::string result = "";
        for (unsigned int id = 0; id < 256; id++)
            result = std::to_string(getbit(id))+result;
        return result;
    }
};


#endif
