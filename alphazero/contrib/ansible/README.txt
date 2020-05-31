This is a minimal setup of Ansible to automatically start simulations.
Next, a few commands.

Run:
ansible-playbook -i inventory.yaml ctrl_run.yaml

Run only certain hosts:
ansible-playbook -i inventory.yaml ctrl_run.yaml --limit euklas,sodalith

A test ping on euklas:
ansible euklas -i inventory.yaml -m ping

If something doesn't work, add task:
- name:
  debug:
    var: vars

Test them all (dry-run):
ansible-playbook -i inventory.yaml --check ctrl_run.yaml

Note regarding setup:
.ssh/config needs to contain an entry for each server. Example for amazonit:
Host amazonit
    Hostname amazonit.cip.ifi.lmu.de
    User yourusername
    Port 22
    ProxyCommand ssh -W %h:%p yourusername@remote.cip.ifi.lmu.de
    ServerAliveInterval 240
Password for ssh can be automated away this way:
ssh-keygen -t rsa -b 2048
ssh-copy-id yourusername@remote.cip.ifi.lmu.de
