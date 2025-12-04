import atexit
import os
import sys
import time
import traceback
import enum
from functools import partial as bind


class Parallel:
    def __init__(self, ctor, strategy):
        self.worker = Worker(bind(self._respond, ctor), strategy, state=True)
        self.callables = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            if name not in self.callables:
                self.callables[name] = self.worker(PMessage.CALLABLE, name)()
            if self.callables[name]:
                return bind(self.worker, PMessage.CALL, name)
            else:
                return self.worker(PMessage.READ, name)()
        except AttributeError:
            raise ValueError(name)

    def __len__(self):
        return self.worker(PMessage.CALL, "__len__")()

    def close(self):
        self.worker.close()

    # --- NEW: Explicit call method for passing commands like set_task ---
    def call(self, name, *args, **kwargs):
        return self.worker(PMessage.CALL, name, *args, **kwargs)()
    # ------------------------------------------------------------------

    @staticmethod
    def _respond(ctor, state, message, name, *args, **kwargs):
        state = state or ctor
        if message == PMessage.CALLABLE:
            assert not args and not kwargs, (args, kwargs)
            # Check if attribute exists and is callable
            attr = getattr(state, name, None)
            if attr is None:
                # Try unwrapping if not found on top level
                real_env = state
                while hasattr(real_env, "_env") or hasattr(real_env, "env"):
                    real_env = getattr(real_env, "_env", getattr(real_env, "env", None))
                    if hasattr(real_env, name):
                        attr = getattr(real_env, name)
                        break
            result = callable(attr)
            
        elif message == PMessage.CALL:
            attr = getattr(state, name, None)
            if attr is None:
                # Try unwrapping for the actual call
                real_env = state
                while hasattr(real_env, "_env") or hasattr(real_env, "env"):
                    real_env = getattr(real_env, "_env", getattr(real_env, "env", None))
                    if hasattr(real_env, name):
                        attr = getattr(real_env, name)
                        break
            
            if attr is None:
                raise AttributeError(f"Method '{name}' not found in environment stack.")
            
            result = attr(*args, **kwargs)
            
        elif message == PMessage.READ:
            assert not args and not kwargs, (args, kwargs)
            result = getattr(state, name)
            
        return state, result


class PMessage(enum.Enum):
    CALLABLE = 2
    CALL = 3
    READ = 4


class Worker:
    initializers = []

    def __init__(self, fn, strategy="thread", state=False):
        if not state:
            fn = lambda s, *args, fn=fn, **kwargs: (s, fn(*args, **kwargs))
        inits = self.initializers
        self.impl = {
            "process": bind(ProcessPipeWorker, initializers=inits),
            "daemon": bind(ProcessPipeWorker, initializers=inits, daemon=True),
        }[strategy](fn)
        self.promise = None

    def __call__(self, *args, **kwargs):
        self.promise and self.promise()  # Raise previous exception if any.
        self.promise = self.impl(*args, **kwargs)
        return self.promise

    def wait(self):
        return self.impl.wait()

    def close(self):
        self.impl.close()


class ProcessPipeWorker:
    def __init__(self, fn, initializers=(), daemon=False):
        import multiprocessing
        import cloudpickle

        self._context = multiprocessing.get_context("spawn")
        self._pipe, pipe = self._context.Pipe()
        fn = cloudpickle.dumps(fn)
        initializers = cloudpickle.dumps(initializers)
        self._process = self._context.Process(
            target=self._loop, args=(pipe, fn, initializers), daemon=daemon
        )
        self._process.start()
        self._nextid = 0
        self._results = {}
        assert self._submit(Message.OK)()
        atexit.register(self.close)

    def __call__(self, *args, **kwargs):
        return self._submit(Message.RUN, (args, kwargs))

    def wait(self):
        pass

    def close(self):
        try:
            self._pipe.send((Message.STOP, self._nextid, None))
            self._pipe.close()
        except (AttributeError, IOError):
            pass  # The connection was already closed.
        try:
            self._process.join(0.1)
            if self._process.exitcode is None:
                try:
                    import signal
                    os.kill(self._process.pid, signal.SIGTERM)
                    time.sleep(0.1)
                except Exception:
                    pass
        except (AttributeError, AssertionError):
            pass

    def _submit(self, message, payload=None):
        callid = self._nextid
        self._nextid += 1
        self._pipe.send((message, callid, payload))
        return Future(self._receive, callid)

    def _receive(self, callid):
        while callid not in self._results:
            try:
                message, callid, payload = self._pipe.recv()
            except (OSError, EOFError):
                raise RuntimeError("Lost connection to worker.")
            if message == Message.ERROR:
                raise Exception(payload)
            assert message == Message.RESULT, message
            self._results[callid] = payload
        return self._results.pop(callid)

    @staticmethod
    def _loop(pipe, function, initializers):
        try:
            callid = None
            state = None
            import cloudpickle

            initializers = cloudpickle.loads(initializers)
            function = cloudpickle.loads(function)
            [fn() for fn in initializers]
            while True:
                if not pipe.poll(0.1):
                    continue  # Wake up for keyboard interrupts.
                message, callid, payload = pipe.recv()
                if message == Message.OK:
                    pipe.send((Message.RESULT, callid, True))
                elif message == Message.STOP:
                    return
                elif message == Message.RUN:
                    args, kwargs = payload
                    state, result = function(state, *args, **kwargs)
                    pipe.send((Message.RESULT, callid, result))
                else:
                    raise KeyError(f"Invalid message: {message}")
        except (EOFError, KeyboardInterrupt):
            return
        except Exception:
            stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            print(f"Error inside process worker: {stacktrace}.", flush=True)
            pipe.send((Message.ERROR, callid, stacktrace))
            return
        finally:
            try:
                pipe.close()
            except Exception:
                pass


class Message(enum.Enum):
    OK = 1
    RUN = 2
    RESULT = 3
    STOP = 4
    ERROR = 5


class Future:
    def __init__(self, receive, callid):
        self._receive = receive
        self._callid = callid
        self._result = None
        self._complete = False

    def __call__(self):
        if not self._complete:
            self._result = self._receive(self._callid)
            self._complete = True
        return self._result


class Damy:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        return lambda: self._env.step(action)

    def reset(self):
        return lambda: self._env.reset()
    
    # --- NEW: Forward call method to inner env ---
def call(self, name, *args, **kwargs):
        # Unwrap manually since Damy doesn't use multiprocessing
        real_env = self._env
        while hasattr(real_env, "_env") or hasattr(real_env, "env"):
            if hasattr(real_env, name):
                method = getattr(real_env, name)
                return method(*args, **kwargs) # <--- EXECUTE IMMEDIATELY
            real_env = getattr(real_env, "_env", getattr(real_env, "env", None))
        
        if hasattr(real_env, name):
            method = getattr(real_env, name)
            return method(*args, **kwargs) # <--- EXECUTE IMMEDIATELY
        raise AttributeError(f"Method {name} not found.")