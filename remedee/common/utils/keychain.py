from getpass import getpass
import os
import sys


def get_env(env_var, service=None, uid=None):
    val = os.environ.get(env_var)
    if val is None and service is not None and uid is not None:
        keychain = KeyChain(service, uid)
        val = keychain.get_pwd(uid)

    return val


class KeyChain:
    def __init__(self, what_for, user=None):
        self.service = what_for
        self.user = os.getenv("USER", default=None) if user is None else user

    def system_security(self, cmd, pwd=""):
        run_cmd = f"security {cmd} -a {self.user} -s {self.service} -w {pwd}"
        stream = os.popen(run_cmd)
        output = stream.read()
        stream.close()
        return output.strip()

    def get_keychain_pwd(self):
        return self.system_security("find-generic-password")

    def set_keychain_pwd(self, pwd):
        self.system_security("add-generic-password", pwd)

    def get_pwd(self, uid):
        pwd = self.get_keychain_pwd()
        if len(pwd) == 0:
            pwd = getpass(prompt=f'Password for {uid}@{self.service}: ')
            self.set_keychain_pwd(pwd)

        return pwd

    def get_uid_pwd(self):
        uid_pwd = self.get_keychain_pwd()
        if len(uid_pwd) == 0:
            uid = input(f'UID for {self.service}: ')
            pwd = getpass(prompt=f'Password for {uid}@{self.service}: ')
            uid_pwd = f"{uid}@{pwd}"
            self.set_keychain_pwd(uid_pwd)
        else:
            uid_pwd_split = uid_pwd.rsplit("@", 1)
            uid = uid_pwd_split[0]
            pwd = uid_pwd_split[1]

        return uid, pwd

if __name__ == "__main__":
    env_var = sys.argv[1]
    service = sys.argv[2]
    uid = sys.argv[3]
    print(get_env(env_var, service, uid))