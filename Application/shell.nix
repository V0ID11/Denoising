{pkgs ? import <nixpkgs> {}}:
pkgs.mkShell {
  buildInputs = [pkgs.python3 pkgs.python3Packages.pip];
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.libGL
    pkgs.libGLU
    pkgs.mesa
    pkgs.glib
    pkgs.dbus
    pkgs.libxkbcommon
    pkgs.xorg.libX11
    pkgs.xorg.libxcb
    pkgs.xorg.libXcomposite
    pkgs.xorg.libXcursor
    pkgs.xorg.libXdamage
    pkgs.xorg.libXext
    pkgs.xorg.libXfixes
    pkgs.xorg.libXi
    pkgs.xorg.libXrandr
    pkgs.xorg.libXrender
    pkgs.xorg.libXtst
    pkgs.xorg.libxshmfence
    pkgs.fontconfig
    pkgs.freetype
    pkgs.wayland
    pkgs.qt6.qtwayland
    pkgs.zlib
    pkgs.zstd
    pkgs.nspr
    pkgs.nss
    pkgs.cups
    pkgs.expat
    pkgs.libdrm
    pkgs.pulseaudio

    # xcb platform plugin extras (from your ldd scan)
    pkgs.xcb-util-cursor
    pkgs.xorg.xcbutilimage
    pkgs.xorg.xcbutilkeysyms
    pkgs.xorg.xcbutilrenderutil
    pkgs.xorg.xcbutilwm # provides libxcb-icccm
    pkgs.xorg.xcbutil # provides libxcb-util

    # GTK theme integration (Qt probes this even if unused)
    pkgs.atk
    pkgs.cairo
    pkgs.gtk3
    pkgs.gdk-pixbuf
    pkgs.pango
    pkgs.harfbuzz

    # Misc image/codec libs Qt image plugins want
    pkgs.brotli
    pkgs.bzip2
    pkgs.libtiff
  ];
}
