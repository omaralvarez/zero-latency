#include <iostream>
#include <fstream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

//#include <EGL/egl.h>
//#include <GL/glew.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <nvpipe.h>

#define PORT 9080
//#define DEBUG

const uint32_t width = 1920;
const uint32_t height = 1080;

#ifdef DEBUG
void captureFramebufferPPM(GLuint framebuffer, uint32_t width, uint32_t height, const std::string& path)
{
    // For verification...

    size_t numBytes = width * height * 3;
    uint8_t* rgb = new uint8_t[numBytes];

    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, rgb);

    std::ofstream outFile;
    outFile.open(path.c_str(), std::ios::binary);

    outFile << "P6" << "\n"
            << width << " " << height << "\n"
            << "255\n";

    outFile.write((char*) rgb, numBytes);

    delete[] rgb;
}
#endif

int main(int argc, char* argv[])
{
    /*
     * General demo setup: Init EGL and OpenGL context
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

    size_t numBytes = serverSendBufferSize;
    nvp_err_t encodeStatus = nvpipe_encode(encoder, serverDeviceBuffer, serverDeviceBufferSize, serverSendBuffer, &numBytes, width, height, NVPIPE_RGBA);
    if (encodeStatus != NVPIPE_SUCCESS)
    {
        std::cerr << "Encode failed: " << std::string(nvpipe_strerror(encodeStatus)) << std::endl;
        return 1;
    }

    std::cout << "osize: " << numBytes << std::endl;

#ifdef DEBUG
    // Export rendered image for verification
    captureFramebufferPPM(0, width, height, "2-client.ppm");
#endif

    int server_fd, new_socket, valread;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
      
    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
    {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }
      
    // Forcefully attaching socket to the port 8080
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT,
                                                  &opt, sizeof(opt)))
    {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons( PORT );
      
    // Forcefully attaching socket to the port 8080
    if (bind(server_fd, (struct sockaddr *)&address, 
                                 sizeof(address))<0)
    {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 3) < 0)
    {
        perror("listen");
        exit(EXIT_FAILURE);
    }
    if ((new_socket = accept(server_fd, (struct sockaddr *)&address, 
                       (socklen_t*)&addrlen))<0)
    {
        perror("accept");
        exit(EXIT_FAILURE);
    }

    // Send the frame size and compressed stream to the consuming side.
    uint32_t size = htonl(numBytes);
    send(new_socket, &size, sizeof(uint32_t), 0);
    send(new_socket, serverSendBuffer, numBytes, 0);

    /*
     * Clean up
     */
    cudaFree(serverDeviceBuffer);
    delete[] serverSendBuffer;
    nvpipe_destroy(encoder);

    return 0;
}