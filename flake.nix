{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
  };
  outputs =
    {
      self,
      nixpkgs,
      utils,
    }:
    utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            pkg-config
            openssl
            gcc
            lld
            binutils
            rustup
          ];

          RUSTUP_TOOLCHAIN = "stable";
          LD = "${pkgs.lld}/bin/ld.lld";
        };
      }
    );
}
