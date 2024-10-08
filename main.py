from github import Github

# Authentication is defined via github.Auth
from github import Auth

from ChangeRephraser import ChangeRephraser
from GithubPR import GithubPR
from ReviewBuddy import ReviewBuddy
from SecurityReviewBuddy import SecurityReviewBuddy
PR_NUMBER = 5
REPO_NAME = "mohmiim/Algo"
TOKEN = open("token", "r").read()
gitPR = GithubPR(REPO_NAME, PR_NUMBER, TOKEN)
change = gitPR.get_all_changes_as_text()
rephraser = ChangeRephraser()
reviewer = ReviewBuddy()
security_reviewer = SecurityReviewBuddy()
description = rephraser.convert(change)
gitPR.update_pr_description(description)
for file, change in gitPR.get_change_for_review_as_text().items():
    print("Reviewing file: ", file)
    comment = reviewer.convert(change)
    #security_comment = security_reviewer.convert(change)
    gitPR.create_comment(file, comment, "ReviewBuddy")
    #gitPR.create_comment(file, security_comment, "SecurityReviewBuddy")
    print("done")
gitPR.close()
