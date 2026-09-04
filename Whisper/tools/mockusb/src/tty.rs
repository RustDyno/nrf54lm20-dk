//! Raw-mode tty handling for the CDC-ACM link.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::path::Path;

use anyhow::{bail, Context, Result};

pub struct Tty {
    f: File,
}

impl Tty {
    pub fn open(path: &Path) -> Result<Tty> {
        // O_NONBLOCK for the open itself: without CLOCAL set yet, opening a
        // tty blocks waiting for carrier. Cleared again once termios is
        // configured.
        let f = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NOCTTY | libc::O_NONBLOCK)
            .open(path)
            .with_context(|| format!("open {}", path.display()))?;
        let fd = f.as_raw_fd();
        unsafe {
            let mut t: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(fd, &mut t) != 0 {
                bail!("tcgetattr {}", path.display());
            }
            libc::cfmakeraw(&mut t);
            // CLOCAL: ignore modem lines. CREAD: enable the receiver. No
            // flow control of any kind -- the protocol is self-framing and
            // the data is binary, so XON/XOFF would corrupt it.
            t.c_cflag |= libc::CLOCAL | libc::CREAD;
            t.c_cflag &= !libc::CRTSCTS;
            // Block until at least one byte, no inter-byte timer; the
            // read loops above handle short reads.
            t.c_cc[libc::VMIN] = 1;
            t.c_cc[libc::VTIME] = 0;
            if libc::tcsetattr(fd, libc::TCSANOW, &t) != 0 {
                bail!("tcsetattr {}", path.display());
            }
            // Between open() and the tcsetattr above, the port is still in
            // the kernel's DEFAULT line discipline: canonical mode with
            // ISIG. Any request byte that lands in that window is mangled,
            // and a request frame carries op=3 at offset 4 -- which is
            // VINTR. Receiving it makes the tty layer FLUSH ITS INPUT
            // QUEUE, silently swallowing the frame's first five bytes and
            // leaving every later read one frame out of step. (Observed as
            // a 512-byte read of near-zeros: the tail of one frame plus the
            // head of the next.)
            //
            // Nothing here can close that window -- the device may have
            // been waiting to send long before this process started -- so
            // discard whatever it produced and let the device's retry
            // deliver a clean frame into a now-raw port.
            libc::tcflush(fd, libc::TCIOFLUSH);
            let fl = libc::fcntl(fd, libc::F_GETFL);
            libc::fcntl(fd, libc::F_SETFL, fl & !libc::O_NONBLOCK);
        }
        Ok(Tty { f })
    }

    pub fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        let mut off = 0;
        while off < buf.len() {
            let n = self.f.read(&mut buf[off..])?;
            if n == 0 {
                bail!("link closed while reading");
            }
            off += n;
        }
        Ok(())
    }

    pub fn write_all(&mut self, buf: &[u8]) -> Result<()> {
        self.f.write_all(buf)?;
        Ok(())
    }
}
