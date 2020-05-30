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
