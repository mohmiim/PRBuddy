from github import Github
# Authentication is defined via github.Auth
from github import Auth


class GithubPR(object):
    def __init__(self, repo_name, pr_number, token):
        self.auth = auth = Auth.Token(token)
        self.gh = Github(auth=auth)
        # Github Enterprise with custom hostname
        # g = Github(base_url="https://{hostname}/api/v3", auth=auth)
        self.repo = self.gh.get_repo(repo_name, lazy=False)
        self.pr = self.repo.get_pull(pr_number)

    def get_all_changes_as_text(self):
        changed_files = self.pr.get_files()
        change = "changes : \n"
        for changed_file in changed_files:
            change += " The file : " + changed_file.filename + " has been"
            if changed_file.status == "added":
                change += " added, "
            elif changed_file.status == "removed":
                change += " removed, "
            elif changed_file.status == "modified":
                change += " modified, "
            change += " the change in github patch format is : \n\t" + changed_file.patch + "\n\n"
        return change

    def get_change_for_review_as_text(self):
        changes = {}
        changed_files = self.pr.get_files()
        for changed_file in changed_files:
            change = " The file : " + changed_file.filename + " has been"
            if changed_file.status == "added":
                change += " added, "
            elif changed_file.status == "removed":
                continue
            elif changed_file.status == "modified":
                change += " modified, "
            change += " the change in github patch format is : \n\t" + changed_file.patch + "\n\n"
            if changed_file.status == "modified":
                change += " The file before the change is : \n" \
                          + self.repo.get_contents(changed_file.filename).decoded_content.decode() \
                          + "\n\n"
            changes[changed_file.filename] = change
        return changes

    def update_pr_description(self, new_description):
        self.pr.edit(body=new_description)

    def close(self):
        self.gh.close()

    def create_comment(self, file, comment, review_by):
        last_commit = self.pr.get_commits()[self.pr.commits - 1]
        self.pr.create_issue_comment(review_by + ":\n" + comment)
