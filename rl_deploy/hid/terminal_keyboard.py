import sys
import threading
import termios
import tty
import atexit
import numpy as np
from dataclasses import dataclass

@dataclass
class TerminalKeyboardConfig:
    """dataclass holding keyboard configuration data"""
    forward_key: str = 'w'
    backward_key: str = 's'
    left_key: str = 'a'
    right_key: str = 'd'
    yaw_left_key: str = 'q'
    yaw_right_key: str = 'e'
    stop_key: str = ' '
    
    delta_forward_velocity: float = 0.5
    delta_lateral_velocity: float = 0.5
    delta_yaw_velocity: float = 0.5
    max_forward_velocity: float = 1.5
    max_backward_velocity: float = 1.5
    max_lateral_velocity: float = 1.5
    max_yaw_velocity: float = 1.0


class TerminalKeyboard:
    def __init__(self, context, config: TerminalKeyboardConfig | None = None, verbose: bool = False, x_vel=0.0, y_vel=0.0, yaw=0.0):
        self.x_vel = x_vel
        self.y_vel = y_vel
        self.yaw = yaw
        self._context = context
        self._config = config if config is not None else TerminalKeyboardConfig()
        self._verbose = verbose
        
        self._stopping = False
        
        # Save original terminal settings for restoration
        self._old_settings = termios.tcgetattr(sys.stdin)
        atexit.register(self._restore_terminal)
        
        self._input_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._input_thread.start()
        
        print("Terminal Keyboard Control Started.")
        print("Controls:")
        print(f"  Forward: {self._config.forward_key}")
        print(f"  Backward: {self._config.backward_key}")
        print(f"  Left: {self._config.left_key}")
        print(f"  Right: {self._config.right_key}")
        print(f"  Yaw L: {self._config.yaw_left_key}")
        print(f"  Yaw R: {self._config.yaw_right_key}")
        print(f"  Stop: {self._config.stop_key} (Space)")

    def _restore_terminal(self):
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
        except Exception:
            pass

    def _read_loop(self):
        try:
            # Set terminal to cbreak mode so we can read one char at a time without Enter
            tty.setcbreak(sys.stdin.fileno())
            while not self._stopping:
                ch = sys.stdin.read(1)
                if not ch:
                    continue
                # Handle input
                ch = ch.lower()
                
                if self._verbose:
                    print(f"\r[DEBUG] Key Pressed: {ch}")
                
                # Update velocity based on key
                if ch == self._config.forward_key:
                    self.x_vel += self._config.delta_forward_velocity
                elif ch == self._config.backward_key:
                    self.x_vel -= self._config.delta_forward_velocity
                elif ch == self._config.left_key:
                    self.y_vel += self._config.delta_lateral_velocity
                elif ch == self._config.right_key:
                    self.y_vel -= self._config.delta_lateral_velocity
                elif ch == self._config.yaw_left_key:
                    self.yaw += self._config.delta_yaw_velocity
                elif ch == self._config.yaw_right_key:
                    self.yaw -= self._config.delta_yaw_velocity
                elif ch == self._config.stop_key:
                    self.x_vel = 0.0
                    self.y_vel = 0.0
                    self.yaw = 0.0
                elif ch == '\x03': # ctrl+c
                    self._stopping = True
                    break

                # Cap velocities
                self.x_vel = float(np.clip(self.x_vel, -self._config.max_backward_velocity, self._config.max_forward_velocity))
                self.y_vel = float(np.clip(self.y_vel, -self._config.max_lateral_velocity, self._config.max_lateral_velocity))
                self.yaw = float(np.clip(self.yaw, -self._config.max_yaw_velocity, self._config.max_yaw_velocity))
                
                #if self._verbose:
                print(f"\rVels -> X: {self.x_vel:.2f}, Y: {self.y_vel:.2f}, Yaw: {self.yaw:.2f}")

        except Exception as e:
            print(f"\r[Error] Keyboard thread: {e}")
        finally:
            self._restore_terminal()

    def listen_loop(self):
        # The main simulation loop repeatedly calls this to execute logic
        if not self._stopping:
            self._context.velocity_cmd = [self.x_vel, self.y_vel, self.yaw]

    def listen(self):
        import time
        while not self._stopping:
            self.listen_loop()
            time.sleep(0.02)

if __name__ == "__main__":
    import time
    
    class MockContext:
        def __init__(self):
            self.velocity_cmd = [0.0, 0.0, 0.0]

    context = MockContext()
    keyboard = TerminalKeyboard(context, verbose=True)
    
    try:
        print("Starting test mode. Press buttons to output velocities, space to stop, Ctrl+C to exit.")
        keyboard.listen()
    except KeyboardInterrupt:
        print("\nExiting...")
        keyboard._stopping = True
