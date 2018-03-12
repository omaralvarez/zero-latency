#include <iostream>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvpipe.h>

#define PORT 9080
//#define DEBUG

const uint32_t width = 1920;
const uint32_t height = 1080;

int main(int argc, char* argv[])
{
    /*
     * Init decoder
     */
    nvpipe* decoder = nvpipe_create_decoder(NVPIPE_H264_NV);
    if (!decoder)
    {
        std::cerr << "ERROR - Decoder creation failed!!!!!" << std::endl;
        return 1;
    }

    size_t clientDeviceBufferSize = width * height * 4;
    void* clientDeviceBuffer;
    if (cudaMalloc(&clientDeviceBuffer, clientDeviceBufferSize) != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory" << std::endl;
        return 1;
    }

    size_t clientReceiveBufferSize = clientDeviceBufferSize; // Reserve enough space for input
    uint8_t* clientReceiveBuffer = new uint8_t[clientReceiveBufferSize];
    memset (clientReceiveBuffer,0,clientReceiveBufferSize);

    struct sockaddr_in address;
    int sock = 0, valread;
    struct sockaddr_in serv_addr;

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        printf("\n Socket creation error \n");
        return -1;
    }
  
    memset(&serv_addr, '0', sizeof(serv_addr));
  
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);
      
    // Convert IPv4 and IPv6 addresses from text to binary form
    if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)<=0) 
    {
        printf("\nInvalid address/ Address not supported \n");
        return -1;
    }
  
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
    {
        printf("\nConnection Failed \n");
        return -1;
    }

    uint32_t size;
    int len = read(sock , &size, sizeof(uint32_t));
    size = ntohl(size);
    std::cout << "Read len: " << len << "Size: " << size << std::endl;


    size_t nbuffer;
    for (nbuffer = 0; nbuffer < size
                           ; nbuffer = std::max(nbuffer, (size_t)size)) { /* Watch out for buffer overflow */
        len = read(sock , clientReceiveBuffer, size);

        /* FIXME: Error checking */

        nbuffer += len;
    }

    std::cout << "Received: " << nbuffer << "/" << size << std::endl;

    size_t imgsz = width * height * 4;
    uint8_t* rgb = (uint8_t*) malloc(imgsz);
    memset (rgb,0,imgsz);

    size_t numBytes = clientReceiveBufferSize;
    nvp_err_t decodeStatus = nvpipe_decode(decoder, clientReceiveBuffer, size, rgb, width, height, NVPIPE_RGBA);
    if (decodeStatus != NVPIPE_SUCCESS)
    {
        std::cerr << "Decode failed: " << std::string(nvpipe_strerror(decodeStatus)) << std::endl;
        return 1;
    }

    std::cout << "Decode finished!" << std::endl;

#ifdef DEBUG
    // Export rendered image for verification
    captureFramebufferPPM(0, width, height, "2-client.ppm");
#endif

    delete[] clientReceiveBuffer;
    cudaFree(clientDeviceBuffer);
    nvpipe_destroy(decoder);

    return 0;
}