# Download models used in tests or for installation

function(download_model url filename hash)
  message(STATUS "Checking for ${filename}")
  file(DOWNLOAD
    ${url}
    ${CMAKE_CURRENT_LIST_DIR}/${filename}
    EXPECTED_HASH ${hash}
    SHOW_PROGRESS
  )
  set(DEFAULT_MODELS "${DEFAULT_MODELS};${CMAKE_CURRENT_LIST_DIR}/${filename}" PARENT_SCOPE)
endfunction()


download_model(
  "https://huggingface.co/Acly/MobileSAM-GGUF/resolve/main/MobileSAM-F16.gguf"
  "MobileSAM-F16.gguf"
  "SHA256=b546366475e3ad744bb2eaf7634df88e9aaf25f6622797d2de300f5a530831f7"
)
download_model(
  "https://huggingface.co/Acly/BiRefNet-GGUF/resolve/main/BiRefNet-lite-F16.gguf"
  "BiRefNet-lite-F16.gguf"
  "SHA256=7b5397a2c98d66677f8f74317774bbeac49dbb321b8a3dc744af913db71d4fa5"
)
download_model(
  "https://huggingface.co/Acly/Depth-Anything-V2-GGUF/resolve/main/Depth-Anything-V2-Small-F16.gguf"
  "Depth-Anything-V2-Small-F16.gguf"
  "SHA256=0f83332d6a8b4375cd7fdcc168f3e3636f474f8e84b0959e903f513aace782f5"
)
download_model(
  "https://huggingface.co/Acly/MIGAN-GGUF/resolve/main/MIGAN-512-places2-F16.gguf"
  "MIGAN-512-places2-F16.gguf"
  "SHA256=3e47592bf716d0dc306f8dc02d4476cfcdaf2c055fa3c3c8e0ced4db775eb64b"
)
if(VISP_TESTS)
  download_model(
    "https://huggingface.co/Acly/Real-ESRGAN-GGUF/resolve/main/RealESRGAN-x4plus_anime-6B-F16.gguf"
    "RealESRGAN-x4plus_anime-6B-F16.gguf"
    "SHA256=730469c5a2269cdef96d0d58aacf87bcf25d7a0d92256685808e6cdce0675c09"
  )
endif()

set(VISP_DEFAULT_MODELS "${DEFAULT_MODELS}" PARENT_SCOPE)
