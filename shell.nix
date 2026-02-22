{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  nativeBuildInputs = [
    pkgs.cudaPackages.cuda_nvcc
    pkgs.cudaPackages.cuda_cudart
  ];

  buildInputs = [
    pkgs.linuxPackages.nvidia_x11
    # apparently order of clang-tools v.s. clang matters: https://blog.kotatsu.dev/posts/2024-04-10-nixpkgs-clangd-missing-headers/
    pkgs.clang-tools
    pkgs.clang
  ];

  packages = with pkgs; [ clang-tools clang ];

  shellHook = ''
    export CUDA_PATH="${pkgs.cudaPackages.cuda_nvcc}"
    export LD_LIBRARY_PATH="${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.cudaPackages.cuda_cudart}/lib:$LD_LIBRARY_PATH"
    nvcc --version
    echo ""
  '';
}
