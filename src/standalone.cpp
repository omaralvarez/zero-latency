#include <cuda.h>
#include <cuda_runtime.h>

#include <cstring>
#include <string>
#include <iostream>
#include <fstream>

#include <nvpipe.h>

int main(int argc, char* argv[])
{
    const uint32_t width = 1920;
    const uint32_t height = 1080;

    /*
     * Server: Init encoder
     */
    const uint64_t bitrate = width * height * 30 * 4 * 0.07; // Kush gauge
    nvpipe* encoder = nvpipe_create_encoder(NVPIPE_H264_NV, bitrate);

    size_t serverDeviceBufferSize = width * height * 4;
    void* serverDeviceBuffer;
    if (cudaMalloc(&serverDeviceBuffer, serverDeviceBufferSize) != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory" << std::endl;
        return 1;
    }

    size_t serverSendBufferSize = serverDeviceBufferSize; // Reserve enough space for encoded output
    uint8_t* serverSendBuffer = new uint8_t[serverSendBufferSize];

    /*
     * Client: Init decoder
     */
    nvpipe* decoder = nvpipe_create_decoder(NVPIPE_H264_NV);

    size_t clientDeviceBufferSize = width * height * 4;
    void* clientDeviceBuffer;
    if (cudaMalloc(&clientDeviceBuffer, clientDeviceBufferSize) != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory" << std::endl;
        return 1;
    }

    size_t clientReceiveBufferSize = clientDeviceBufferSize; // Reserve enough space for input
    uint8_t* clientReceiveBuffer = new uint8_t[clientReceiveBufferSize];

    /*
     * Server: Grab frame and encode
     */

    size_t numBytes = serverSendBufferSize;
    nvp_err_t encodeStatus = nvpipe_encode(encoder, serverDeviceBuffer, width * height * 4, serverSendBuffer, &numBytes, width, height, NVPIPE_RGBA);
    if (encodeStatus != NVPIPE_SUCCESS)
    {
        std::cerr << "Encode failed: " << std::string(nvpipe_strerror(encodeStatus)) << std::endl;
        return 1;
    }


    /*
     * Network transfer (e.g. over TCP socket)
     */
    // ... Send buffer size, buffer data, width, height, etc. (whatever you need)
    memcpy(clientReceiveBuffer, serverSendBuffer, numBytes); // Dummy


    /*
     * Client: Decode to OpenGL texture
     */
    nvp_err_t decodeStatus = nvpipe_decode(decoder, clientReceiveBuffer, numBytes, clientDeviceBuffer, width, height, NVPIPE_RGBA);
    if (decodeStatus != NVPIPE_SUCCESS)
    {
        std::cerr << "Decode failed: " << std::string(nvpipe_strerror(decodeStatus)) << std::endl;
        return 1;
    }

    /*
     * Clean up
     */
    cudaFree(serverDeviceBuffer);
    delete[] serverSendBuffer;
    nvpipe_destroy(encoder);

    delete[] clientReceiveBuffer;
    cudaFree(clientDeviceBuffer);
    nvpipe_destroy(decoder);

    return 0;
}